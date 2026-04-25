import os
import re
from pathlib import Path
from bs4 import BeautifulSoup

RAW_DIR = Path("data/texts_raw/gretil")

OUT_DIR = Path("data/texts_clean/gretil")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_html_file(input_path: Path, output_path: Path):
    """Read one GRETIL file, strip HTML, clean whitespace, save as .txt."""
    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    soup = BeautifulSoup(content, "html.parser")
    text = soup.get_text()

    text = re.sub(r"\n\s*\n+", "\n\n", text)
    text = text.strip()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)


def main():
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            if file.lower().endswith((".htm", ".html", ".txt", ".xml")):
                src = Path(root) / file
                rel = src.relative_to(RAW_DIR)
                dest = OUT_DIR / rel.with_suffix(".txt")

                print(f"[+] Cleaning {src}")
                clean_html_file(src, dest)

    print("\n[✓] Finished cleaning GRETIL texts.")
    print(f"    Cleaned files are under: {OUT_DIR}")


if __name__ == "__main__":
    main()
