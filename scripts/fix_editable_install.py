"""Repair a hidden project .pth link that Python 3.13 intentionally skips on macOS."""

import os
from pathlib import Path
import site
import stat
import sys


def main():
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("Run this with a project virtual-environment interpreter, not global Python")
    source = str(Path(__file__).resolve().parents[1] / "src")
    for directory in site.getsitepackages():
        directory = Path(directory)
        if not directory.is_relative_to(Path(sys.prefix)):
            continue
        for path in directory.glob("__editable__.s1small-*.pth"):
            if path.read_text().strip() != source:
                raise ValueError("Editable link does not point to this project's source directory")
            flags = getattr(path.stat(), "st_flags", 0)
            hidden = getattr(stat, "UF_HIDDEN", 0)
            if hidden and flags & hidden:
                os.chflags(path, flags & ~hidden)
                print(f"Cleared hidden flag on project editable link: {path.name}")


if __name__ == "__main__":
    main()
