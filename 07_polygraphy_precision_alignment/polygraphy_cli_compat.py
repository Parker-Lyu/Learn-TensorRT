#!/usr/bin/env python3
"""Run Polygraphy with local compatibility patches for this lesson environment."""

from __future__ import annotations

import sys

import numpy as np


# Polygraphy 0.49.x still references np.unicode_, which NumPy 2 removed.
# Keeping the patch here avoids changing the global Python environment.
if not hasattr(np, "unicode_"):
    np.unicode_ = np.str_  # type: ignore[attr-defined]


def main() -> int:
    from polygraphy.tools._main import main as polygraphy_main

    return int(polygraphy_main())


if __name__ == "__main__":
    raise SystemExit(main())
