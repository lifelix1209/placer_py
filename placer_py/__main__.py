"""Entry point for `python -m placer_py`.

The console script `placer-py` and `python3 -m placer_py.main` both already
worked; `python -m placer_py` did not, which is the spelling most people try
first when the package is installed but the scripts directory is not on PATH.
"""

from __future__ import annotations

import sys

from placer_py.main import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
