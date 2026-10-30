#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
README_PATH="$ROOT_DIR/README.md"
START_MARKER="<!-- TREE:START -->"
END_MARKER="<!-- TREE:END -->"

python3 - "$ROOT_DIR" "$README_PATH" "$START_MARKER" "$END_MARKER" <<'PY'
from __future__ import annotations

import re
import sys
from pathlib import Path


root = Path(sys.argv[1]).resolve()
readme_path = Path(sys.argv[2]).resolve()
start_marker = sys.argv[3]
end_marker = sys.argv[4]

exclude_names = {".git", ".venv", ".pytest_cache", ".DS_Store"}
abbreviated_dirs = {
    Path("data/ligo_cache"),
    Path("data/pantheon"),
    Path("results/chains"),
    Path("results/figures"),
    Path("results/tables"),
}


def iter_children(directory: Path):
    entries = []
    for entry in directory.iterdir():
        if entry.name in exclude_names:
            continue
        if entry.name == "__pycache__":
            continue
        entries.append(entry)
    entries.sort(key=lambda entry: (not entry.is_dir(), entry.name.lower()))
    return entries


def build_tree_lines(rel_dir: Path = Path("."), prefix: str = "") -> list[str]:
    absolute_dir = (root / rel_dir).resolve() if rel_dir != Path(".") else root
    lines: list[str] = []
    children = iter_children(absolute_dir)
    for index, child in enumerate(children):
        is_last = index == len(children) - 1
        connector = "└── " if is_last else "├── "
        name = f"{child.name}/" if child.is_dir() else child.name
        lines.append(f"{prefix}{connector}{name}")

        if not child.is_dir() or child.is_symlink():
            continue

        child_rel = rel_dir / child.name if rel_dir != Path(".") else Path(child.name)
        next_prefix = f"{prefix}{'    ' if is_last else '│   '}"
        if child_rel in abbreviated_dirs:
            lines.append(f"{next_prefix}└── ...")
            continue

        lines.extend(build_tree_lines(child_rel, next_prefix))
    return lines


tree = "\n".join([f"{root.name}/", *build_tree_lines()])
replacement_block = f"{start_marker}\n```text\n{tree}\n```\n{end_marker}"

readme = readme_path.read_text(encoding="utf-8")
pattern = re.compile(re.escape(start_marker) + r".*?" + re.escape(end_marker), re.DOTALL)

if not pattern.search(readme):
    raise SystemExit(
        f"Could not find markers in {readme_path}. Expected {start_marker} ... {end_marker}."
    )

updated = pattern.sub(replacement_block, readme, count=1)
readme_path.write_text(updated, encoding="utf-8")
print(f"Updated tree in {readme_path}")
PY
