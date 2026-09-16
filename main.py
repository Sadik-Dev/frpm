#!/usr/bin/env python
"""Entry point: ``python main.py "https://www.youtube.com/watch?v=..."``

Works whether or not the project has been ``pip install -e .``'d, by
falling back to putting ``src/`` on ``sys.path`` directly.
"""
from __future__ import annotations

import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from frpm.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
