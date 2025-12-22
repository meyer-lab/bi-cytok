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
        # Look for Python code block markers
        if lines[i].strip().startswith("```{python}"):
            code_start = i + 1
            i += 1
            code_lines = []
            while i < len(lines) and not lines[i].strip().startswith("```"):
                code_lines.append(lines[i])
                i += 1

            end_line = i

            if code_lines:
                # Filter out lines that aren't valid Python
                filtered_lines = [
                    line
                    for line in code_lines
                    if not line.lstrip().startswith("%")  # IPython magic
                    and not line.lstrip().startswith("!")  # Shell commands
                    and not line.lstrip().startswith("#|")  # Quarto options
                ]
                code = "\n".join(filtered_lines)
                if code.strip():
                    blocks.append((code_start, end_line, code, code_lines))
        i += 1

    return blocks


def adjust_output_paths(output: str, filepath: Path, code_start: int) -> str:
    """Adjust ruff output to show actual file path instead of temp path."""
    lines = output.split("\n")
    result = []

    for line in lines:
        # Replace temp file path with actual file path
        # Match the pattern: /tmp/tmpXXXXXX.py (includes letters, numbers, underscores, hyphens)
        match = re.search(r"/tmp/tmp[a-zA-Z0-9_-]+\.py", line)
        if match:
            temp_file = match.group(0)
            line = line.replace(temp_file, str(filepath))
            # Find line number reference like "1:4" and adjust it
            line_match = re.search(r"(\d+):(\d+)", line)
            if line_match:
                temp_line_num = int(line_match.group(1))
                col_num = line_match.group(2)
                actual_line_num = temp_line_num + code_start - 1
                line = re.sub(
                    r"\d+:\d+",
                    f"{actual_line_num}:{col_num}",
                    line,
                    count=1,
                )
        elif match := re.match(r"^(\s+\|)?\s+(\d+)\s+\|", line):
            # Adjust line numbers in the code context display
            temp_line_num = int(match.group(2))
            actual_line_num = temp_line_num + code_start - 1
            line = re.sub(r"^\s+(\d+)\s+\|", f"{actual_line_num} |", line)
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

        # Create a temporary Python file for ruff to check
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            # First, check if there are any issues (without fixing)
            check_result = subprocess.run(
                ["ruff", "check", tmp_path],
                capture_output=True,
                text=True,
            )

            if check_result.returncode != 0:
                issues_found = True

            if check_only:
                # In check-only mode, display the check output and don't fix
                if check_result.stdout or check_result.stderr:
                    output = check_result.stdout + check_result.stderr
                    if (
                        "All checks passed!" not in output
                        or check_result.returncode != 0
                    ):
                        adjusted_output = adjust_output_paths(
                            output, filepath, code_start
                        )
                        print(adjusted_output, end="")
            else:
                # Now run with --fix to fix the issues
                result = subprocess.run(
                    ["ruff", "check", "--fix", tmp_path],
                    capture_output=True,
                    text=True,
                )

                # Always display output (even when fixing, to show what was fixed or what remains)
                # But filter out "All checks passed!" messages that are just noise
                if result.stdout or result.stderr:
                    output = result.stdout + result.stderr
                    # Skip "All checks passed!" if there are no actual issues to report
                    if (
                        "All checks passed!" not in output
                        or check_result.returncode != 0
                    ):
                        adjusted_output = adjust_output_paths(
                            output, filepath, code_start
                        )
                        print(adjusted_output, end="")

                # Read back the corrected code
                corrected = Path(tmp_path).read_text()
                # Remove the trailing newline we added for ruff, then split
                corrected_lines = corrected.rstrip("\n").split("\n")

                # Merge corrected code with original non-Python lines (Quarto options, etc.)
                merged_lines = merge_code_blocks(original_lines, corrected_lines)

                # Replace the code lines (keeping the block markers)
                lines = lines[:code_start] + merged_lines + lines[code_end:]
        finally:
            Path(tmp_path).unlink()

    # Write back the fixed content (only if not in check-only mode)
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
        # Ensure code ends with newline for ruff
        if not code.endswith("\n"):
            code = code + "\n"

        # Create a temporary Python file for ruff to format
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            # First check if formatting would make changes
            check_result = subprocess.run(
                ["ruff", "format", "--check", tmp_path],
                capture_output=True,
                text=True,
            )

            if check_result.returncode != 0:
                issues_found = True
                # Print that formatting is needed (only once per file)
                if not has_printed_reformat_msg:
                    print(f"would reformat {filepath}")
                    has_printed_reformat_msg = True

            # Run ruff format
            result = subprocess.run(
                ["ruff", "format", tmp_path],
                capture_output=True,
                text=True,
            )

            # Display any output (ruff format usually has minimal output)
            if result.stdout or result.stderr:
                output = result.stdout + result.stderr
                # Filter out "1 file reformatted" messages that are per-block noise
                if "1 file" not in output:
                    adjusted_output = adjust_output_paths(output, filepath, code_start)
                    print(adjusted_output, end="")

            # Read back the formatted code
            formatted = Path(tmp_path).read_text()
            # Remove the trailing newline we added for ruff, then split
            formatted_lines = formatted.rstrip("\n").split("\n")

            # Merge formatted code with original non-Python lines (Quarto options, etc.)
            merged_lines = merge_code_blocks(original_lines, formatted_lines)

            # Replace the code lines (keeping the block markers)
            lines = lines[:code_start] + merged_lines + lines[code_end:]
        finally:
            Path(tmp_path).unlink()

    # Write back the formatted content (only if not in check mode)
    if not check_only:
        filepath.write_text("\n".join(lines))

    return 1 if issues_found else 0


def merge_code_blocks(
    original_lines: list[str], corrected_lines: list[str]
) -> list[str]:
    """Merge corrected code with original non-Python lines.

    Preserves Quarto options (#|), IPython magics (%), and shell commands (!)
    at their original positions, while updating the actual Python code.
    """
    merged = []
    corrected_idx = 0

    for orig_line in original_lines:
        stripped = orig_line.lstrip()
        # Preserve non-Python lines in their original positions
        if (
            stripped.startswith("#|")
            or stripped.startswith("%")
            or stripped.startswith("!")
        ):
            merged.append(orig_line)
        else:
            # Use corrected Python code line
            if corrected_idx < len(corrected_lines):
                merged.append(corrected_lines[corrected_idx])
                corrected_idx += 1

    # Add any remaining corrected lines (shouldn't normally happen)
    while corrected_idx < len(corrected_lines):
        merged.append(corrected_lines[corrected_idx])
        corrected_idx += 1

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
        if args.command == "check":
            check_only = getattr(args, "check", False)
            if check_and_fix_qmd_file(filepath, check_only=check_only) != 0:
                exit_code = 1
        elif args.command == "format":
            check_only = getattr(args, "check", False)
            if format_qmd_file(filepath, check_only=check_only) != 0:
                exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
