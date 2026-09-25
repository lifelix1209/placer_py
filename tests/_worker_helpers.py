"""Functions a spawned worker can import by name, for tests of the process pool.

A spawned process re-imports whatever it runs by module name, so these cannot
live in a test module (whose import name depends on pytest's import mode).
`tests/` is on the path of both runners, and a spawned child inherits it.
"""

import os
import signal


def square(x):
    return x * x


def square_unless_three(x):
    """Dies the way a crashing C extension does: by signal, with no traceback."""
    if x == 3:
        os.kill(os.getpid(), signal.SIGKILL)
    return x * x
