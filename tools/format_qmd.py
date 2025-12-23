#!/usr/bin/env python3
"""Lint and format Python code blocks in Quarto markdown files."""

import argparse
import re
import subprocess
import sys
import tempfile
from pathlib import Path


def extract_python_blocks(qmd_content: str) -> list[tuple[int, int, str, list[str]]]:
    """Extract Python code blocks from Quarto markdown content.

    Returns a list of (start_line, end_line, code, original_lines) tuples for each Python block.
    Filters out:
    - IPython magic commands (%, !!)
    - Quarto code chunk options (#|)
    But preserves them to rewrite later.
    """
    blocks = []
    lines = qmd_content.split("\n")
    i = 0
    while i < len(lines):
        if lines[i].strip().startswith("```{python}"):
            code_start = i + 1
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1

            if code_lines:
                filtered_lines = [
                    line
                    for line in code_lines
                    if not line.lstrip().startswith(("%", "!", "#|"))
                ]
                code = "\n".join(filtered_lines)
                if code.strip():
                    blocks.append((code_start, i, code, code_lines))
        i += 1

    return blocks


def adjust_output_paths(output: str, filepath: Path, code_start: int) -> str:
    """Replace temp file paths with actual paths and adjust line numbers for interpretable check output."""
    lines = output.split("\n")
    result = []

    for line in lines:
        match = re.search(r"/tmp/tmp[a-zA-Z0-9_-]+\.py", line)
        if match:
            line = line.replace(match.group(0), str(filepath))
            if line_match := re.search(r"(\d+):(\d+)", line):
                actual_line = int(line_match.group(1)) + code_start - 1
                line = re.sub(
                    r"\d+:\d+",
                    f"{actual_line}:{line_match.group(2)}",
                    line,
                    count=1,
                )
        elif match := re.match(r"^(\s+\|)?\s+(\d+)\s+\|", line):
            actual_line = int(match.group(2)) + code_start - 1
            line = re.sub(r"^\s+(\d+)\s+\|", f"{actual_line} |", line)
        result.append(line)

    return "\n".join(result)


def check_and_fix_qmd_file(filepath: Path, check_only: bool = False) -> int:
    """Check and fix linting issues in a Quarto markdown file.

    Args:
        filepath: Path to the Quarto file
        check_only: If True, only check for issues without fixing them

    Returns 0 if no issues, 1 if issues found (even after fixing).
    """
    content = filepath.read_text()
    blocks = extract_python_blocks(content)
    if not blocks:
        return 0

    issues_found = False
    lines = content.split("\n")

    # Process blocks in reverse order to avoid line number shifting
    for code_start, code_end, code, original_lines in reversed(blocks):
        # Ensure code ends with newline for ruff
        if not code.endswith("\n"):
            code = code + "\n"

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            check_result = subprocess.run(
                ["ruff", "check", tmp_path],
                capture_output=True,
                text=True,
            )

            if check_result.returncode != 0:
                issues_found = True

            if check_only:
                if check_result.stdout or check_result.stderr:
                    output = check_result.stdout + check_result.stderr
                    if (
                        "All checks passed!" not in output
                        or check_result.returncode != 0
                    ):
                        print(adjust_output_paths(output, filepath, code_start), end="")
            else:
                result = subprocess.run(
                    ["ruff", "check", "--fix", tmp_path],
                    capture_output=True,
                    text=True,
                )

                if result.stdout or result.stderr:
                    output = result.stdout + result.stderr
                    # Skip "All checks passed!" if there are no issues to report
                    if (
                        "All checks passed!" not in output
                        or check_result.returncode != 0
                    ):
                        print(adjust_output_paths(output, filepath, code_start), end="")

                corrected = Path(tmp_path).read_text()
                corrected_lines = corrected.rstrip("\n").split("\n")
                merged_lines = merge_code_blocks(original_lines, corrected_lines)
                lines = lines[:code_start] + merged_lines + lines[code_end:]
        finally:
            Path(tmp_path).unlink()

    if not check_only:
        filepath.write_text("\n".join(lines))

    return 1 if issues_found else 0


def format_qmd_file(filepath: Path, check_only: bool = False) -> int:
    """Format Python code blocks in a Quarto markdown file.

    Args:
        filepath: Path to the Quarto file
        check_only: If True, only check if formatting would change (don't apply changes)

    Returns 0 if no issues, 1 if issues found (even after fixing).
    """
    content = filepath.read_text()
    blocks = extract_python_blocks(content)
    if not blocks:
        return 0

    issues_found = False
    has_printed_reformat_msg = False
    lines = content.split("\n")

    # Process blocks in reverse order to avoid line number shifting
    for code_start, code_end, code, original_lines in reversed(blocks):
        code = code + "\n" if not code.endswith("\n") else code

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            check_result = subprocess.run(
                ["ruff", "format", "--check", tmp_path],
                capture_output=True,
                text=True,
            )

            if check_result.returncode != 0:
                issues_found = True
                if not has_printed_reformat_msg:
                    print(f"would reformat {filepath}")
                    has_printed_reformat_msg = True

            if not check_only:
                result = subprocess.run(
                    ["ruff", "format", tmp_path],
                    capture_output=True,
                    text=True,
                )

                if result.stdout or result.stderr:
                    output = result.stdout + result.stderr
                    if "1 file" not in output:
                        print(adjust_output_paths(output, filepath, code_start), end="")

                formatted = Path(tmp_path).read_text()
                formatted_lines = formatted.rstrip("\n").split("\n")
                merged_lines = merge_code_blocks(original_lines, formatted_lines)
                lines = lines[:code_start] + merged_lines + lines[code_end:]
        finally:
            Path(tmp_path).unlink()

    if not check_only:
        filepath.write_text("\n".join(lines))

    return 1 if issues_found else 0


def merge_code_blocks(
    original_lines: list[str], corrected_lines: list[str]
) -> list[str]:
    """Merge corrected code with Quarto directives/magics."""
    merged = []
    corrected_idx = 0

    for orig_line in original_lines:
        stripped = orig_line.lstrip()
        if stripped.startswith(("#|", "%", "!")):
            merged.append(orig_line)
        elif corrected_idx < len(corrected_lines):
            merged.append(corrected_lines[corrected_idx])
            corrected_idx += 1

    merged.extend(corrected_lines[corrected_idx:])
    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Lint and format Python code blocks in Quarto markdown files"
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Check and fix linting issues
    check_parser = subparsers.add_parser(
        "check", help="Check and fix linting issues (like ruff check --fix)"
    )
    check_parser.add_argument(
        "--check",
        action="store_true",
        help="Check for issues without fixing them (like ruff check)",
    )
    check_parser.add_argument(
        "files", nargs="+", help="Quarto files or directories to lint"
    )

    # Format code
    format_parser = subparsers.add_parser(
        "format", help="Format code (like ruff format)"
    )
    format_parser.add_argument(
        "--check",
        action="store_true",
        help="Check if formatting would make changes without applying them",
    )
    format_parser.add_argument(
        "files", nargs="+", help="Quarto files or directories to format"
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    files_to_process = []
    for file_arg in args.files:
        path = Path(file_arg)
        if path.is_dir():
            files_to_process.extend(path.glob("*.qmd"))
        elif path.is_file():
            files_to_process.append(path)

    exit_code = 0
    for filepath in sorted(files_to_process):
        result = None
        if args.command == "check":
            result = check_and_fix_qmd_file(filepath, check_only=args.check)
        elif args.command == "format":
            result = format_qmd_file(filepath, check_only=args.check)

        if result != 0:
            exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
