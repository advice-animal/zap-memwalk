"""Frida JS agent injected into the target process.

setup() probes the target's pymalloc internals once (py_version, pool_size).
collect() scans all rw- ranges for pool_header signatures — no GIL, no
           knowledge of _obmalloc_state required, works on stripped builds.
readPool() returns raw block data for the block-level hex view.
reprBlock() acquires the GIL, bumps refcount, and calls PyObject_Repr safely.
"""

from __future__ import annotations

_AGENT_JS = r"""
'use strict';

// ── cached state (set during setup) ──────────────────────────────────────────
let _poolSize  = 16384;
let _arenaSize = 1048576;

// ── symbol / NativeFunction helpers (pattern from zap-allocator) ─────────────
function sym(name) {
    if (typeof Module.findGlobalExportByName === 'function') {
        const p = Module.findGlobalExportByName(name);
        if (p && !p.isNull()) return p;
    }
    for (const mod of Process.enumerateModules()) {
        try {
            const p = mod.getExportByName(name);
            if (p && !p.isNull()) return p;
        } catch (e) {}
    }
    return null;
}

function nfn(name, ret, args) {
    const p = sym(name);
    if (!p) throw new Error('symbol not found: ' + name);
    return new NativeFunction(p, ret, args);
}

function makeSetupRunner() {
    const ensure  = nfn('PyGILState_Ensure',  'int',  []);
    const release = nfn('PyGILState_Release', 'void', ['int']);
    const runStr  = nfn('PyRun_SimpleString', 'int',  ['pointer']);
    return function run(code) {
        const state = ensure();
        const ret   = runStr(Memory.allocUtf8String(code));
        release(state);
        return ret;
    };
}

// Read one Python integer attribute from __main__ via PyLong_AsVoidPtr.
// Returns a NativePointer whose integer value equals the attribute,
// or null pointer if missing / not an int.
function readIntAttr(run, attrName) {
    const ensure    = nfn('PyGILState_Ensure',       'int',     []);
    const release   = nfn('PyGILState_Release',      'void',    ['int']);
    const addMod    = nfn('PyImport_AddModule',      'pointer', ['pointer']);
    const getAttr   = nfn('PyObject_GetAttrString',  'pointer', ['pointer', 'pointer']);
    const asVoidPtr = nfn('PyLong_AsVoidPtr',        'pointer', ['pointer']);
    const decRef    = nfn('Py_DecRef',               'void',    ['pointer']);

    const cMain = Memory.allocUtf8String('__main__');
    const cAttr = Memory.allocUtf8String(attrName);

    const state   = ensure();
    const mainMod = addMod(cMain);
    const obj     = getAttr(mainMod, cAttr);
    let result    = ptr('0x0');
    if (!obj.isNull()) {
        const p = asVoidPtr(obj);
        if (!p.isNull()) result = p;
        decRef(obj);
    }
    release(state);
    return result;
}

// ── free-list walker (no GIL needed) ─────────────────────────────────────────
function walkFreeList(poolPtr, maxBlocks) {
    const addrs = [];
    let cur;
    try { cur = poolPtr.add(8).readPointer(); } catch (e) { return addrs; }
    let limit = 0;
    while (cur && !cur.isNull() && limit < maxBlocks) {
        addrs.push(cur.toString());          // hex string preserves 64-bit precision
        try { cur = cur.readPointer(); } catch (e) { break; }
        limit++;
    }
    return addrs;
}

// ── pool-header scan over ALL rw- ranges ─────────────────────────────────────
// Validation uses the exact maxnextoffset = POOL_SIZE - block_size invariant.
// Scans in 4 MiB chunks to bound peak memory usage from readByteArray.
function scanAllPools() {
    const POOL_SIZE     = _poolSize;
    const POOL_MASK     = POOL_SIZE - 1;
    const POOL_OVERHEAD = 48;
    const CHUNK_SIZE    = 4 * 1024 * 1024;

    const pools = [];

    for (const range of Process.enumerateRanges('rw-')) {
        if (range.size < POOL_SIZE) continue;

        const rangeBase = parseInt(range.base.toString());
        const rangeEnd  = rangeBase + range.size;

        for (let chunkStart = rangeBase; chunkStart < rangeEnd; chunkStart += CHUNK_SIZE) {
            const chunkSize = Math.min(CHUNK_SIZE, rangeEnd - chunkStart);
            let buf;
            try {
                buf = ptr('0x' + chunkStart.toString(16)).readByteArray(chunkSize);
            } catch (e) { continue; }

            const dv = new DataView(buf);
            // First pool-aligned offset within this chunk
            const firstOff = (POOL_SIZE - (chunkStart & POOL_MASK)) & POOL_MASK;

            for (let off = firstOff; off + POOL_SIZE <= chunkSize; off += POOL_SIZE) {
                const szidx = dv.getUint32(off + 36, true);
                if (szidx >= 32) continue;

                const blockSize     = (szidx + 1) * 16;
                const maxnextoffset = dv.getUint32(off + 44, true);
                // Exact invariant from CPython obmalloc.c: POOL_SIZE - block_size
                if (maxnextoffset !== POOL_SIZE - blockSize) continue;

                const refCount   = dv.getUint32(off + 0,  true);
                const nextoffset = dv.getUint32(off + 40, true);
                const totalBlocks = (POOL_SIZE - POOL_OVERHEAD) / blockSize;

                if (refCount > totalBlocks) continue;
                if (nextoffset > POOL_SIZE)  continue;

                const paddr  = chunkStart + off;
                const poolPtr = ptr('0x' + paddr.toString(16));
                const maxBl  = Math.ceil(totalBlocks) + 2;

                const arenaIndex = dv.getUint32(off + 32, true);
                const freeAddrs  = walkFreeList(poolPtr, maxBl);

                pools.push({
                    address:      paddr.toString(16),
                    arenaIndex,
                    szidx,
                    refCount,
                    nextoffset,
                    maxnextoffset,
                    freeAddrs,
                });
            }
        }
    }
    return pools;
}

// ── RPC exports ───────────────────────────────────────────────────────────────
rpc.exports = {

    setup: function () {
        let run;
        try { run = makeSetupRunner(); }
        catch (e) { return {ok: false, error: e.message}; }

        // Retrieve Python version via ctypes — no struct offset guessing needed.
        const ret = run(`
import ctypes as _ct, sys as _sys, __main__ as _m
_m._mw_pymaj = _sys.version_info.major
_m._mw_pymin = _sys.version_info.minor
`);
        if (ret !== 0) return {ok: false, error: 'setup Python snippet failed'};

        const pyMaj = parseInt(readIntAttr(run, '_mw_pymaj').toString());
        const pyMin = parseInt(readIntAttr(run, '_mw_pymin').toString());

        return {ok: true, poolSize: _poolSize, arenaSize: _arenaSize,
                pyMajor: pyMaj, pyMinor: pyMin};
    },

    collect: function () {
        try {
            return {ok: true, pools: scanAllPools()};
        } catch (e) {
            return {ok: false, error: e.message};
        }
    },

    // Refresh pool header + free list + raw bytes for the block view.
    // NOTE: must be named readPool (camelCase) to match Frida's Python→JS mapping.
    // raw bytes are returned as a base64 string because ArrayBuffer inside an RPC
    // dict doesn't serialize reliably across the Frida bridge.
    readPool: function (poolAddrHex) {
        const p = ptr('0x' + poolAddrHex);
        try {
            const szidx         = p.add(36).readU32();
            const refCount      = p.add(0).readU32();
            const nextoffset    = p.add(40).readU32();
            const maxnextoffset = p.add(44).readU32();
            const blockSize     = (szidx + 1) * 16;
            const totalBlocks   = (_poolSize - 48) / blockSize;
            const maxBl         = Math.ceil(totalBlocks) + 2;
            const freeAddrs     = walkFreeList(p, maxBl);
            const rawBuf        = p.readByteArray(_poolSize);
            // hex-encode so it survives the JSON RPC bridge
            const rawHex        = Array.from(new Uint8Array(rawBuf),
                                      b => b.toString(16).padStart(2, '0')).join('');
            return {ok: true, szidx, refCount, nextoffset, maxnextoffset, freeAddrs, rawHex};
        } catch (e) {
            return {ok: false, error: e.message};
        }
    },

    // Safely repr a live block.
    // 1) Read ob_type + tp_name without GIL (memory reads only).
    // 2) Sanity-check refcount.
    // 3) Acquire GIL, Py_IncRef, PyObject_Repr, Py_DecRef, release GIL.
    // NOTE: must be named reprBlock (camelCase) to match Frida's Python→JS mapping.
    reprBlock: function (blockAddrHex) {
        const blockPtr = ptr('0x' + blockAddrHex);

        let typeName = '?';
        try {
            const obType = blockPtr.add(8).readPointer();   // PyObject.ob_type
            const tpName = obType.add(24).readPointer();    // PyTypeObject.tp_name
            typeName = tpName.readUtf8String();
        } catch (e) {}

        let refcnt = 0;
        try { refcnt = blockPtr.add(0).readU32(); } catch (e) {}
        if (refcnt === 0 || refcnt > 0x7fffffff) {
            return {ok: false, typeName, error: 'implausible refcount — block may be free'};
        }

        const ensure   = nfn('PyGILState_Ensure',  'int',     []);
        const release  = nfn('PyGILState_Release', 'void',    ['int']);
        const incRef   = nfn('Py_IncRef',           'void',    ['pointer']);
        const decRef   = nfn('Py_DecRef',           'void',    ['pointer']);
        const reprFn   = nfn('PyObject_Repr',       'pointer', ['pointer']);
        const asUtf8   = nfn('PyUnicode_AsUTF8',    'pointer', ['pointer']);
        const clearErr = nfn('PyErr_Clear',         'void',    []);

        const state = ensure();
        incRef(blockPtr);
        let reprStr = null;
        let errMsg  = null;
        try {
            const reprObj = reprFn(blockPtr);
            if (!reprObj.isNull()) {
                const cstr = asUtf8(reprObj);
                reprStr = cstr.isNull() ? null : cstr.readUtf8String();
                decRef(reprObj);
            } else {
                clearErr();
                errMsg = 'PyObject_Repr returned NULL';
            }
        } catch (e) {
            errMsg = e.message;
        } finally {
            decRef(blockPtr);
            release(state);
        }

        return reprStr !== null
            ? {ok: true, typeName, reprStr}
            : {ok: false, typeName, error: errMsg || 'repr failed'};
    },

    // Bulk-resolve ob_type pointers → tp_name strings (no GIL needed).
    // Input: array of hex strings (ob_type pointer values from raw block bytes).
    // Returns: object mapping each input hex string to its tp_name (or '?').
    // NOTE: must be named resolveTypeNames (camelCase) to match Frida's mapping.
    resolveTypeNames: function (typeAddrHexList) {
        const result = {};
        for (const addrHex of typeAddrHexList) {
            try {
                const typePtr = ptr('0x' + addrHex);
                const tpNamePtr = typePtr.add(24).readPointer();
                result[addrHex] = tpNamePtr.readUtf8String();
            } catch (e) {
                result[addrHex] = '?';
            }
        }
        return result;
    },

    // Bulk-resolve addresses → {module, offset, symbol} or null.
    // Uses Process.findModuleByAddress (no GIL, cross-platform) for module lookup
    // and DebugSymbol.fromAddress (best-effort) for symbol names.
    // NOTE: must be named symbolizeAddresses (camelCase) for Frida's Python→JS mapping.
    symbolizeAddresses: function(addrHexList) {
        const result = {};
        for (const addrHex of addrHexList) {
            try {
                const p = ptr('0x' + addrHex);
                // DebugSymbol.fromAddress uses dladdr() which covers __DATA segments
                // correctly on macOS (unlike Module.size which only covers __TEXT).
                const sym = DebugSymbol.fromAddress(p);
                const modName = sym && sym.moduleName && sym.moduleName.length > 0
                    ? sym.moduleName : null;
                if (!modName) { result[addrHex] = null; continue; }

                // Compute byte offset from the module's load address.
                let offset = 0;
                const mod = Process.findModuleByName(modName);
                if (mod) offset = parseInt(p.sub(mod.base).toString());

                // sym.name may be an address string ("0x...") when no symbol is found.
                const rawName = sym.name || '';
                const symbolName = (rawName.length > 0 && !rawName.startsWith('0x'))
                    ? rawName : null;

                result[addrHex] = { module: modName, offset: offset, symbol: symbolName };
            } catch (_) {
                result[addrHex] = null;
            }
        }
        return result;
    },
};
"""
