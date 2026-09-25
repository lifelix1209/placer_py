"""Entry point for `python -m placer`.

The console script `placer` and `python3 -m placer.main` both already
worked; `python -m placer` did not, which is the spelling most people try
first when the package is installed but the scripts directory is not on PATH.
"""

from __future__ import annotations

import sys

from placer.main import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
