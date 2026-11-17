"""Entry point of a chain worker process.

`quivers.inference.mcmc._chain_process.run_worker_processes` starts each
worker as ``python -m quivers.inference.mcmc._chain_entry``. No other module
imports this one, so it is executed once, as the worker's main module.
"""

from __future__ import annotations

import sys

from quivers.inference.mcmc._chain_process import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
