"""The exclusive-create-then-read-only write shared by every store that
checks a file in once and never overwrites it in place: FileRunStore,
FileExtractionRecordStore, screening.gold_set.write_gold_set, and
screening.eval_roles.write_evaluation_role.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TextIO


def write_once(path: Path, write_body: Callable[[TextIO], object], *, mode: int = 0o644) -> Path:
    """Creates `path` exclusively - raising the built-in FileExistsError if
    it already exists, so a completed write is never silently overwritten -
    calls `write_body` with the open file handle, then makes the file
    read-only. A `write_body` that raises partway through leaves no file
    behind: the partial write is removed before the exception propagates,
    so a failed write never permanently blocks `path` for a retry.
    """
    fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, mode)
    try:
        with os.fdopen(fd, "w") as fh:
            write_body(fh)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    path.chmod(0o444)
    return path
