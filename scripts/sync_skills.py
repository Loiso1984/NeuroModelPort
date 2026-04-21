#!/usr/bin/env python3
"""Mirror canonical skills into editor shadow folders.

Source of truth: .ai/skills/
Targets:
  .codex/skills/
  .cursor/skills/
  .kiro/skills/
  .pi/skills/
  .trae/skills/
  .trae-cn/skills/
"""

from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path


SHADOW_FOLDERS = (
    ".codex",
    ".cursor",
    ".kiro",
    ".pi",
    ".trae",
    ".trae-cn",
)


@dataclass
class SyncStats:
    copied: int = 0
    skipped: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sync SKILL.md files from .ai/skills to editor shadow folders.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned operations without writing files.",
    )
    return parser.parse_args()


def iter_skill_files(source_dir: Path) -> list[Path]:
    return sorted(source_dir.rglob("SKILL.md"))


def sync_file(src: Path, dst: Path, dry_run: bool) -> bool:
    if dst.exists() and src.read_bytes() == dst.read_bytes():
        return False
    if not dry_run:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
    return True


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    source_root = repo_root / ".ai" / "skills"
    if not source_root.is_dir():
        raise SystemExit(f"Missing canonical skills directory: {source_root}")

    skill_files = iter_skill_files(source_root)
    if not skill_files:
        print(f"No SKILL.md files found under {source_root}")
        return 0

    total = SyncStats()
    for folder in SHADOW_FOLDERS:
        target_root = repo_root / folder / "skills"
        folder_stats = SyncStats()

        for src in skill_files:
            relative = src.relative_to(source_root)
            dst = target_root / relative
            changed = sync_file(src, dst, dry_run=args.dry_run)
            if changed:
                folder_stats.copied += 1
            else:
                folder_stats.skipped += 1

        action = "would copy" if args.dry_run else "copied"
        print(
            f"{folder}: {action} {folder_stats.copied} file(s), "
            f"skipped {folder_stats.skipped} up-to-date file(s)"
        )
        total.copied += folder_stats.copied
        total.skipped += folder_stats.skipped

    final_action = "would copy" if args.dry_run else "copied"
    print(
        f"Done: {final_action} {total.copied} file(s), "
        f"skipped {total.skipped} file(s)."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
