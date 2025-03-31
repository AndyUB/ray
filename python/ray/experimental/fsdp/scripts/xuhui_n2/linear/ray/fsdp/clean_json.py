import os
import re
from argparse import ArgumentParser


def clean_and_replace_file(file_path: str, encoding: str = "utf-8") -> None:
    """
    Read a file ignoring Unicode decode errors, then replace the original file
    with a clean version.

    Args:
        file_path (str): Path to the file with potential encoding errors
        encoding (str): Encoding to use (default: 'utf-8')
    """

    # Create a temporary file name
    temp_file = file_path + ".tmp"

    # Read the original file, ignoring decode errors
    with open(file_path, "r", encoding=encoding, errors="ignore") as input_file:
        readable_content = input_file.read()

    clean_content = re.sub(r"[\x00-\x09\x0B\x0C\x0E-\x1F\x7F]", "", readable_content)

    # Write the clean content to a temporary file
    with open(temp_file, "w", encoding=encoding) as output_file:
        output_file.write(clean_content)

    # Replace the original file with the cleaned one
    os.replace(temp_file, file_path)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--json_path", type=str, required=True)
    json_path = parser.parse_args().json_path

    try:
        clean_and_replace_file(json_path)
        print(f"File cleaned and replaced successfully.")
    except Exception as e:
        print(f"Error: {e}")
