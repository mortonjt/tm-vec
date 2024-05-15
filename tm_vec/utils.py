#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from pathlib import Path
from dataclasses import dataclass, field
from pysam import FastxFile
from typing import Dict, Generator, Tuple, List


@dataclass
class SessionTree:
    """
    Creates a model for session dir.
    root/
        checkpoints/
        logs/
        params.json
        dataset_indices.pkl
    """
    root: Path

    def __post_init__(self):
        self.root = Path(self.root)

        self.root.mkdir(exist_ok=True, parents=True)
        self.checkpoints.mkdir(exist_ok=True, parents=True)
        self.logs.mkdir(exist_ok=True, parents=True)

    @property
    def params(self):
        return self.root / "params.json"

    @property
    def checkpoints(self):
        return self.root / "checkpoints"

    @property
    def indices(self):
        return self.root / "dataset_indices.pkl"

    @property
    def logs(self):
        return self.root / "logs/"

    @property
    def last_ckpt(self):
        return self.checkpoints / "last.ckpt"

    @property
    def best_ckpt(self):
        if (self.checkpoints / "best.ckpt").exists():
            self.checkpoints / "best.ckpt"
        return self.last_ckpt

    def dump_indices(self, indices):
        with open(self.indices, 'wb') as pk:
            pickle.dump(indices, pk)


def load_fasta_as_dict(fasta_file: str, max_seqs = None) -> Dict[str, str]:
    """
    Load FASTA file as dict of headers to sequences

    Args:
        fasta_file (str): Path to FASTA file. Can be compressed.

    Returns:sq
        Dict[str, str]: Dictionary of FASTA entries sorted by length.
    """

    seqs_dict = {}
    with FastxFile(fasta_file) as f:
        for i, entry in enumerate(f):
            seqs_dict[entry.name] = entry.sequence
            if max_seqs and i == max_seqs:
                break

    return seqs_dict


def create_batched_sequence_datasest(
    sequences: Dict[str, str], max_tokens_per_batch: int = 1024
) -> Generator[Tuple[List[str], List[str]], None, None]:

    batch_headers, batch_sequences, num_tokens = [], [], 0
    for header, seq in sequences.items():
        if (len(seq) + num_tokens > max_tokens_per_batch) and num_tokens > 0:
            yield batch_headers, batch_sequences
            batch_headers, batch_sequences, num_tokens = [], [], 0
        batch_headers.append(header)
        batch_sequences.append(seq)
        num_tokens += len(seq)

    yield batch_headers, batch_sequences


if __name__ == '__main__':
    pass
