"""Pre-commit hook: block commits when any `changelogs/**/*.md` file exceeds the line limit.

When the live `changelogs/changelog.md` outgrows the limit, the workflow is to rotate
it into `changelogs/archives/changelog_<YYYYMMDD>.md` (filename uses the most recent
date inside the file) and start a fresh `changelogs/changelog.md` below the limit.
"""

from __future__ import annotations

from pathlib import Path
import sys

LIMIT = 300


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    changelogs = root / "changelogs"
    if not changelogs.is_dir():
        return 0
    over: list[tuple[str, int]] = []
    for path in sorted(changelogs.rglob("*.md")):
        n = len(path.read_text().splitlines())
        if n > LIMIT:
            over.append((str(path.relative_to(root)), n))
    if not over:
        return 0
    print(f"changelog file(s) exceed {LIMIT} lines:", file=sys.stderr)
    for path, n in over:
        print(f"  {path}: {n} lines", file=sys.stderr)
    print(
        "Rotate the file into changelogs/archives/changelog_<YYYYMMDD>.md "
        "(YYYYMMDD = newest date inside the file) and start a fresh "
        "changelogs/changelog.md.",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
