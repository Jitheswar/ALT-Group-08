"""screening.fileio.write_once is the exclusive-create-then-read-only
primitive every checked-in-once store (FileRunStore, FileExtractionRecordStore,
gold_set.write_gold_set, eval_roles.write_evaluation_role) shares. Tested
directly here rather than only indirectly through each of those callers.
"""

from __future__ import annotations

import stat
from pathlib import Path

import pytest

from screening.fileio import write_once


def test_write_once_writes_the_body_and_makes_the_file_read_only(tmp_path: Path):
    path = tmp_path / "record.json"

    result = write_once(path, lambda fh: fh.write("hello"))

    assert result == path
    assert path.read_text() == "hello"
    assert not stat.S_IMODE(path.stat().st_mode) & 0o222


def test_write_once_never_overwrites_an_existing_file(tmp_path: Path):
    path = tmp_path / "record.json"
    write_once(path, lambda fh: fh.write("first"))

    with pytest.raises(FileExistsError):
        write_once(path, lambda fh: fh.write("second"))

    assert path.read_text() == "first"


def test_a_write_body_that_raises_leaves_no_file_behind(tmp_path: Path):
    path = tmp_path / "record.json"

    def boom(fh) -> None:
        fh.write("partial")
        raise ValueError("boom")

    with pytest.raises(ValueError):
        write_once(path, boom)

    assert not path.exists()

    # And the path is free to try again.
    write_once(path, lambda fh: fh.write("retry"))
    assert path.read_text() == "retry"
