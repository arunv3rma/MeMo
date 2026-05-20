import json
from pathlib import Path
from typing import Literal





dataset_root = Path(__file__).resolve().parent / "_data"


def load_book_long():
    return (dataset_root / "grammar_book_for_claude_long.txt").read_text()


def load_book_medium():
    return (dataset_root / "grammar_book_for_claude_medium.txt").read_text()


def load_book_full():
    return (dataset_root / "grammar_book.txt").read_text()


def load_book_full_tex():
    return (dataset_root / "grammar_book.tex").read_text()


def load_wordlist():
    return json.loads((dataset_root / "wordlist.json").read_text())


def load_test_ek():
    data = json.loads((dataset_root / "test_examples_ek.json").read_text())[1:]
    assert len(data) == 50
    return data


def load_test_ke():
    data = json.loads((dataset_root / "test_examples_ke.json").read_text())[1:]
    assert len(data) == 50
    return data


def load_train_examples():
    data = json.loads((dataset_root / "train_examples.json").read_text())[1:]
    return data


def wordlist_to_lines(wordlist: dict[str, list[str] | str]) -> list:
    return (
        [
            f"{source}: {','.join(target) if isinstance(target, list) else target}"
            for (source, target) in wordlist.items()
        ]
    )

