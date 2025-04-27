import re


def clean_control_chars(file_path: str, output_path: str) -> str:
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    # Remove all control characters including tabs (0x09),
    # but keep newlines (0x0A) and carriage returns (0x0D)
    cleaned = re.sub(r"[\x00-\x1F\x7F]", "", content)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(cleaned)

    print(f"Cleaned file written to {output_path}")
    return cleaned


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Clean control characters from a file")
    parser.add_argument("file_path", help="Path to the input file")
    parser.add_argument("--output", help="Output path (default: input_file.clean)")

    args = parser.parse_args()
    output_path = args.output if args.output else args.file_path + ".clean"

    clean_control_chars(args.file_path, output_path)
