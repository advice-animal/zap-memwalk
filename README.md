# zap-memwalk

This is a TUI to examine the memory of a running Python interpreter.

Views:

1. Size classes
2. Pools
3. Blocks

You can hit `/` to search for an address that you got from `id(v)` or
`hex(id(v))` which will jump to the pool with the right entry selected.  You
can press `r` to attempt a repr of the item, which only works for live objects.

Navigation is `j/k/o` or `up/down/enter`.  Page with `n/p` or `pgdn/pgup`.

# Version Compat

This library is compatile with Python 3.10+, but should be linted under the
newest stable version.

# Versioning

This library follows [meanver](https://meanver.org/) which basically means
[semver](https://semver.org/) along with a promise to rename when the major
version changes.

# License

zap-memwalk is copyright [Tim Hatch](https://timhatch.com/), and licensed under
the MIT license.  See the `LICENSE` file for details.
