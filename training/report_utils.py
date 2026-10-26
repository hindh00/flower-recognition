"""Shared helper for appending/replacing a titled section in docs/results.md
Used by every explainability script"""

from pathlib import Path


def append_results_section(results_path: Path, marker: str, lines: list[str]) -> None:
    new_section = "\n".join([marker, ""] + lines).rstrip() + "\n"
    existing = results_path.read_text() if results_path.exists() else ""
    if marker in existing:
        before = existing.split(marker)[0].rstrip() + "\n\n"
    else:
        before = existing.rstrip() + "\n\n" if existing else ""
    results_path.write_text(before + new_section)
