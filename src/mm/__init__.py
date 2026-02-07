"""Minimal entrypoint for python-mini-metro."""

from __future__ import annotations

import os
import runpy
import sys


def run() -> None:
    """Run the vendored python-mini-metro game."""
    package_dir = os.path.dirname(__file__)
    env_dir = os.path.abspath(os.path.join(package_dir, "game"))
    main_path = os.path.join(env_dir, "main.py")

    if env_dir not in sys.path:
        sys.path.insert(0, env_dir)

    runpy.run_path(main_path, run_name="__main__")


if __name__ == "__main__":
    run()
