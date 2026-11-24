"""Portable manifests and atomic outputs."""

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import tempfile

import numpy as np


@contextmanager
def atomic(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as file:
        temporary = Path(file.name)
    try:
        yield temporary
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    with atomic(path) as temporary:
        temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_jsonl(path, rows):
    with atomic(path) as temporary:
        with temporary.open("w") as file:
            for row in rows:
                file.write(json.dumps(row, allow_nan=False) + "\n")


def read_jsonl(path):
    with Path(path).open() as file:
        return [json.loads(line) for line in file if line.strip()]


def save_npz(path, **arrays):
    with atomic(path) as temporary, temporary.open("wb") as file:
        np.savez_compressed(file, **arrays)


def save_checkpoint(path, state):
    import torch

    with atomic(path) as temporary:
        torch.save(state, temporary)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return dict(data)
