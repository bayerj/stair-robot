"""CLI entry point for trajectory optimization package."""

import defopt

from .optimize import optimize
from .playback import playback
from .tune import tune_run_best, tune_search, tune_show_best

if __name__ == "__main__":
    defopt.run(
        [
            optimize,
            playback,
            tune_search,
            tune_run_best,
            tune_show_best,
        ]
    )
