from pathlib import Path

def read_all_txt_files(folder: str) -> list[str]:
    """
    Read all .txt files in the specified folder and return their contents as a list of strings.
    """
    folder_path = Path(folder)
    if not folder_path.is_dir():
        raise ValueError(f"The provided path '{folder}' is not a valid directory.")

    # non-recursive: only files directly in the folder
    paths = sorted(Path(folder).glob("*.txt"))   # sort for stable order
    return [p.read_text(encoding="utf-8", errors="ignore") for p in paths]