"""Run from the repository root, even before installing the project."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from s1small.environment import main


if __name__ == "__main__":
    main()

