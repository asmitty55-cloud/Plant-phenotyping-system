"""Run the curated plant phenotyping validation dataset from a source checkout."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pt.validation import main


if __name__ == "__main__":
    raise SystemExit(main())
