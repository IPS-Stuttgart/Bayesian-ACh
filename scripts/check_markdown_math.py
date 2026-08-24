"""Reject Markdown math forms that GitHub's sanitizer can mangle.

GitHub supports fenced ``math`` blocks for display math and the backtick-delimited
inline form ``$`...`$``.  This checker keeps project documentation on those two
forms and deliberately keeps headings free of math.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SKIP_PARTS = {".git", ".venv", "build", "dist"}


def markdown_files() -> list[Path]:
    return sorted(
        path
        for path in ROOT.rglob("*.md")
        if not any(part in SKIP_PARTS for part in path.parts)
    )


def check_file(path: Path) -> list[str]:
    errors: list[str] = []
    in_fence = False
    fence_marker = ""
    fence_language = ""

    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        stripped = line.lstrip()

        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if not in_fence:
                in_fence = True
                fence_marker = marker
                fence_language = stripped[3:].strip().split(maxsplit=1)[0] if stripped[3:].strip() else ""
                if fence_language == "math" and len(line) != len(stripped):
                    errors.append(
                        f"{path.relative_to(ROOT)}:{line_number}: math fences must not be indented"
                    )
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
                fence_language = ""
            continue

        if in_fence:
            continue

        rel = path.relative_to(ROOT)

        if stripped.startswith("#") and "$" in stripped:
            errors.append(f"{rel}:{line_number}: keep headings free of math")

        for forbidden, description in (
            (r"\operatorname", r"use standard operators or \mathrm instead of \operatorname"),
            (r"\(", r"use GitHub inline math $`...`$ instead of \(...\)"),
            (r"\)", r"use GitHub inline math $`...`$ instead of \(...\)"),
            (r"\[", r"use fenced math blocks instead of \[...\]"),
            (r"\]", r"use fenced math blocks instead of \[...\]"),
            ("$$", "use fenced ```math blocks instead of $$...$$"),
        ):
            if forbidden in line:
                errors.append(f"{rel}:{line_number}: {description}")

        # Remove the supported inline delimiters before looking for stray dollars.
        remainder = line.replace("$`", "").replace("`$", "").replace(r"\$", "")
        if "$" in remainder:
            errors.append(
                f"{rel}:{line_number}: use GitHub sanitizer-safe inline math $`...`$"
            )

    if in_fence:
        errors.append(f"{path.relative_to(ROOT)}: unclosed fenced code block")

    return errors


def main() -> int:
    errors = [error for path in markdown_files() for error in check_file(path)]
    if errors:
        print("Markdown math compatibility check failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("Markdown math compatibility check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
