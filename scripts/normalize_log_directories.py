#!/usr/bin/env python
"""Normalize experiment logs to one paper-named directory per method.

Run with ``--apply`` after reviewing the default dry-run report. Redundant or
lower-quality files with the same filename are preserved outside ``logs/`` in
``run_output/log_duplicates`` rather than overwritten.
"""

from __future__ import annotations

import argparse
import ast
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.logging_utils import canonical_log_dir


LOG_ROOT = ROOT / "logs"
RUN_OUTPUT = ROOT / "run_output"

# Directories produced by older runs. Values are the paper-facing destination
# directories used by the workbook and by trainer.py going forward.
LOG_DIR_ALIASES = {
    "Full Fine-Tuning": "Full Fine-Tuning",
    "Finetune": "Full Fine-Tuning",
    "SimpleCIL": "SimpleCIL",
    "APER": "APER",
    "adam_adapter": "APER",
    "L2P": "L2P",
    "DualPrompt": "DualPrompt",
    "CODA-Prompt": "CODA-Prompt",
    "LAE": "LAE",
    "lae": "LAE",
    "InfLoRA": "InfLoRA",
    "inflora": "InfLoRA",
    "EASE": "EASE",
    "BiLoRA": "BiLoRA",
    "bilora": "BiLoRA",
    "SD-LoRA": "SD-LoRA",
    "SD-LoRA-IN21K": "SD-LoRA",
    "sdlora": "SD-LoRA",
    "CL-LoRA": "CL-LoRA",
    "cllora": "CL-LoRA",
    "iCaRL": "iCaRL",
    "FOSTER": "FOSTER",
    "DER": "DER",
}

CURVE_RE = re.compile(r"CNN top1 curve: (\[[^\r\n]+\])")
AVERAGE_RE = re.compile(r"Average Accuracy \(CNN\): ([0-9.]+)")


def quality(path: Path):
    """Prefer logs with more finished tasks, then a reported average, then mtime."""
    text = path.read_text(errors="ignore")
    curves = CURVE_RE.findall(text)
    curve_length = -1
    if curves:
        try:
            curve_length = len(ast.literal_eval(curves[-1]))
        except (SyntaxError, ValueError):
            pass
    return (curve_length, bool(AVERAGE_RE.findall(text)), path.stat().st_mtime)


def unique_archive_path(method: str, source: Path):
    archive = RUN_OUTPUT / "log_duplicates" / method
    stem, suffix = source.stem, source.suffix
    candidate = archive / "{}__{}{}".format(source.parent.name, stem, suffix)
    index = 1
    while candidate.exists():
        candidate = archive / "{}__{}__{}{}".format(source.parent.name, stem, index, suffix)
        index += 1
    return candidate


def move(path: Path, destination: Path, apply: bool):
    print("{} -> {}".format(path.relative_to(ROOT), destination.relative_to(ROOT)))
    if apply:
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))


def collect_method_logs():
    grouped = defaultdict(list)
    for directory, method in LOG_DIR_ALIASES.items():
        source = LOG_ROOT / directory
        if source.is_dir():
            grouped[method].extend(source.glob("*.log"))

    archive = LOG_ROOT / "第二轮A800"
    if archive.is_dir():
        for source in archive.iterdir():
            if not source.is_dir() or source.name.startswith(".") or source.name == "launchers":
                continue
            method = LOG_DIR_ALIASES.get(source.name, canonical_log_dir(source.name))
            grouped[method].extend(source.glob("*.log"))
    return grouped


def normalize_logs(apply: bool):
    grouped = collect_method_logs()
    moved = archived = 0
    for method in sorted(grouped):
        destination_dir = LOG_ROOT / method
        by_name = defaultdict(list)
        for path in grouped[method]:
            by_name[path.name].append(path)
        for filename, paths in sorted(by_name.items()):
            selected = max(paths, key=quality)
            destination = destination_dir / filename
            for duplicate in paths:
                if duplicate == selected:
                    continue
                archive_path = unique_archive_path(method, duplicate)
                move(duplicate, archive_path, apply)
                archived += 1
            if selected != destination:
                move(selected, destination, apply)
                moved += 1

    print("selected_moves={} archived_duplicates={}".format(moved, archived))


def relocate_non_logs(apply: bool):
    candidates = []
    for path in LOG_ROOT.rglob("*"):
        if path.is_file() and path.suffix != ".log":
            candidates.append(path)
    for path in sorted(candidates):
        destination = RUN_OUTPUT / "progress" / path.relative_to(LOG_ROOT)
        move(path, destination, apply)


def remove_empty_directories(apply: bool):
    for path in sorted((item for item in LOG_ROOT.rglob("*") if item.is_dir()), key=lambda item: len(item.parts), reverse=True):
        if path == LOG_ROOT or any(path.iterdir()):
            continue
        print("remove empty {}".format(path.relative_to(ROOT)))
        if apply:
            path.rmdir()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="perform the normalization")
    args = parser.parse_args()
    normalize_logs(args.apply)
    relocate_non_logs(args.apply)
    remove_empty_directories(args.apply)
    print("mode={}".format("apply" if args.apply else "dry-run"))


if __name__ == "__main__":
    main()
