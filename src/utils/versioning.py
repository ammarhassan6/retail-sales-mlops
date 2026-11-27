"""Helpers that record WHICH code and WHICH data produced a model.

- git_commit(): the current Git commit id (code version)
- file_md5():   the MD5 hash of a data file. DVC uses the same MD5 for a
                single file, so this value matches the hash in dvc.lock
                (data version).
"""

import hashlib
import os
import subprocess
from pathlib import Path


def git_commit() -> str:
    # Inside Docker there is no .git folder, so CI can pass it as an env variable
    if os.getenv("GIT_COMMIT"):
        return os.environ["GIT_COMMIT"]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def last_commit_for(path: str) -> str:
    """Commit id of the last change to a folder, e.g. the Feast feature repo
    ("which version of the feature definitions was used?")."""
    try:
        result = subprocess.run(
            ["git", "log", "-1", "--format=%h", "--", path],
            capture_output=True, text=True, check=True,
        )
        return result.stdout.strip() or "uncommitted"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def file_md5(path: str | Path) -> str:
    md5 = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            md5.update(chunk)
    return md5.hexdigest()
