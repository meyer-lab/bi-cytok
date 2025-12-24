#!/usr/bin/env python3
"""Lint and format Python code blocks in Quarto markdown files."""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path


def extract_python_blocks(qmd_content: str) -> list[tuple[int, int, str, list[str]]]:
    """
    Extract Python code blocks from Quarto markdown content.

    Args:
        qmd_content: The code content of the Quarto markdown file as a string.

    Returns a list of (start_line, end_line, code, original_lines) tuples for each Python block.
    Filters out:
    - IPython magic commands (%, !)
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


def check_and_fix_qmd_file(filepath: Path, check_only: bool = False) -> int:
    """
    Check and fix linting issues in a Quarto markdown file.

    Args:
        filepath: Path to the Quarto file
        check_only: If True, only check for issues without fixing them

    Returns 0 if no issues, 1 if issues were found and fixed, 2 if issues were found
        but not fixed.
    """
    content = filepath.read_text()
    blocks = extract_python_blocks(content)
    if not blocks:
        return 0
    file_id = filepath.stem + "_"

    issues_found = False
    unfixed_issues = False
    lines = content.split("\n")

    # Process blocks in reverse order to avoid line number shifting
    for code_start, code_end, code, original_lines in reversed(blocks):
        # Ensure code ends with newline for ruff
        if not code.endswith("\n"):
            code = code + "\n"

        with tempfile.NamedTemporaryFile(
            mode="w", prefix=file_id, suffix=".py", delete=False
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            # Check for linting issues
            check_result = subprocess.run(
                ["ruff", "check", tmp_path],
                capture_output=True,
                text=True,
            )
            if check_result.returncode != 0:
                issues_found = True
            if check_only and (check_result.stdout or check_result.stderr):
                output = check_result.stdout + check_result.stderr
                if "All checks passed!" not in output or check_result.returncode != 0:
                    print(
                        f"Linting issue(s) found in file: {filepath}, chunk lines: {code_start}-{code_end}"
                    )
                    print(output)

            # Attempt to fix linting issues
            if not check_only:
                result = subprocess.run(
                    ["ruff", "check", "--fix", tmp_path],
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    unfixed_issues = True

                # Print issues from ruff output
                if result.stdout or result.stderr:
                    output = result.stdout + result.stderr
                    if "All checks passed!" not in output or result.returncode != 0:
                        print(
                            f"Linting issue(s) found in file: {filepath}, chunk lines: {code_start}-{code_end}"
                        )
                        print(output)

                # Read changes, merge with Quarto magics, and append updates to lines
                corrected = Path(tmp_path).read_text()
                corrected_lines = corrected.rstrip("\n").split("\n")
                merged_lines = merge_code_blocks(original_lines, corrected_lines)
                lines = lines[:code_start] + merged_lines + lines[code_end:]
        finally:
            Path(tmp_path).unlink()

    # Write back changes if not in check-only mode
    if not check_only:
        filepath.write_text("\n".join(lines))
        if unfixed_issues:
            return 2
        elif issues_found:
            return 1
        else:
            return 0

    return 1 if issues_found else 0


def format_qmd_file(filepath: Path, check_only: bool = False) -> int:
    """
    Format Python code blocks in a Quarto markdown file.

    Args:
        filepath: Path to the Quarto file
        check_only: If True, only check if formatting would change

    Returns 0 if no issues, 1 if issues were found and fixed, 2 if issues were found
        but not fixed.
    """
    content = filepath.read_text()
    blocks = extract_python_blocks(content)
    if not blocks:
        return 0
    file_id = filepath.stem + "_"

    issues_found = False
    format_failed = False
    lines = content.split("\n")

    # Process blocks in reverse order to avoid line number shifting
    for code_start, code_end, code, original_lines in reversed(blocks):
        code = code + "\n" if not code.endswith("\n") else code

        with tempfile.NamedTemporaryFile(
            mode="w", prefix=file_id, suffix=".py", delete=False
        ) as tmp:
            tmp.write(code)
            tmp_path = tmp.name

        try:
            # Check for formatting issues
            check_result = subprocess.run(
                ["ruff", "format", "--check", tmp_path],
                capture_output=True,
                text=True,
            )
            if check_result.returncode != 0:
                issues_found = True
            if check_only and (check_result.stdout or check_result.stderr):
                output = check_result.stdout + check_result.stderr
                if "already" not in output:
                    print(
                        f"Formatting issue(s) found in file: {filepath}, chunk lines: {code_start}-{code_end}"
                    )
                    print(output)

            # Attempt to fix formatting issues
            if not check_only:
                result = subprocess.run(
                    ["ruff", "format", tmp_path],
                    capture_output=True,
                    text=True,
                )

                if result.returncode != 0:
                    format_failed = True

                if result.stdout or result.stderr:
                    output = result.stdout + result.stderr
                    if "unchanged" not in output:
                        print(
                            f"Formatting issue(s) found in file: {filepath}, chunk lines: {code_start}-{code_end}"
                        )
                        print(output)

                formatted = Path(tmp_path).read_text()
                formatted_lines = formatted.rstrip("\n").split("\n")
                merged_lines = merge_code_blocks(original_lines, formatted_lines)
                lines = lines[:code_start] + merged_lines + lines[code_end:]
        finally:
            Path(tmp_path).unlink()

    if not check_only:
        filepath.write_text("\n".join(lines))
        if format_failed:
            return 2
        elif issues_found:
            return 1
        else:
            return 0

    return 1 if issues_found else 0


def merge_code_blocks(
    original_lines: list[str], corrected_lines: list[str]
) -> list[str]:
    """
    Merge corrected code with Quarto directives/magics.

    Args:
        original_lines: The original lines of the code block.
        corrected_lines: The corrected/formatted lines of the code block.

    Returns the original directive lines combined with corrected code.
    """
    merged = []
    corrected_idx = 0

    for orig_line in original_lines:
        stripped = orig_line.lstrip()
        # Preserve lines starting with Quarto directives or magics, which will cause
        #   problems if directives are present anywhere but the start of a chunk
        if stripped.startswith(("#|", "%", "!")):
            merged.append(orig_line)
        elif corrected_idx < len(corrected_lines):
            merged.append(corrected_lines[corrected_idx])
            corrected_idx += 1

    # Append remaining corrected lines (beyond the length of original code)
    merged.extend(corrected_lines[corrected_idx:])
    return merged


def main():
    """
    Main function to parse arguments and run linting/formatting on Quarto files.

    Returns an exit code indicating success (code 0) or degree of failture (code 1 
        indicating fixed/fixable issues, 2 indicating infixable issues).
    """
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
            result = check_and_fix_qmd_file(filepath, check_only=args.check)
        elif args.command == "format":
            result = format_qmd_file(filepath, check_only=args.check)
        else:
            continue

        if result == 1:
            exit_code = 1
        elif result == 2:
            exit_code = 2

    if exit_code == 0:
        print("All checks passed!")
    elif exit_code == 1 and args.check:
        print("Some checks failed.")
    elif exit_code == 1 and not args.check:
        print("Some checks failed, but were successfully fixed.")
    else:
        print("Some checks failed and could not be fixed.")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
