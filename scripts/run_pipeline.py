"""CLI wrapper for the plant phenotyping pipeline."""

import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from pt.pipeline.run_pipeline import main


if __name__ == "__main__":
    main()
