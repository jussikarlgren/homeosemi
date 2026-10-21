"""Put the bundled homeosemi and jussipyutils repos on sys.path.

Both repos use flat, top-level module imports (e.g. `import sparsevectors`,
`import hyperdimensionalsemanticspace`), so their directories must be importable.
Import this module before importing anything from either repo.
"""
import os
import sys

_here = os.path.dirname(os.path.abspath(__file__))

# jussipyutils supplies sparsevectors / logger / languagemodel;
# homeosemi supplies hyperdimensionalsemanticspace.
for _sub in ("jussipyutils", "homeosemi"):
    _p = os.path.join(_here, _sub)
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)
