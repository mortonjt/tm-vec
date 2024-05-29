#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Generator, List, Tuple

import numpy as np
import torch
from pysam import FastxFile


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


def load_fasta_as_dict(fasta_file: str,
                       sort: bool = True,
                       max_len: int = None) -> Dict[str, str]:
    """
    Load FASTA file as dict of headers to sequences

    Args:
        fasta_file (str): Path to FASTA file. Can be compressed.
        sorted (bool): Sort sequences by length.
        max_len (int): Maximum length of sequences to include.

    Returns:sq
        Dict[str, str]: Dictionary of FASTA entries sorted by length.
    """

    seqs_dict = {}
    with FastxFile(fasta_file) as f:
        for i, entry in enumerate(f):
            seqs_dict[entry.name] = entry.sequence

    if sort:
        seqs_dict = dict(sorted(seqs_dict.items(), key=lambda x: len(x[1])))

    if max_len:
        seqs_dict = {k: v for k, v in seqs_dict.items() if len(v) <= max_len}

    return seqs_dict


def create_batched_sequence_datasest(
    sequences: Dict[str, str],
    max_tokens_per_batch: int = 1024
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


# Generate random proteins
def generate_proteins(n_prots):

    PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
    np.random.seed(42)
    proteins = []
    for _ in range(n_prots):
        prot = "".join(
            np.random.choice(list(PROTEIN_ALPHABET),
                             size=np.random.randint(20, 100)))
        proteins.append(prot)
    return proteins


# Predict the TM-score for a pair of proteins (inputs are TM-Vec embeddings)
def cosine_similarity(output_seq1: torch.Tensor,
                      output_seq2: torch.Tensor) -> torch.Tensor:
    """
    Calculate the cosine similarity between two protein embeddings

    Args:
        output_seq1 (torch.Tensor): Protein embedding for sequence 1
        output_seq2 (torch.Tensor): Protein embedding for sequence 2

    Returns:
        torch.Tensor: Cosine similarity between the two embeddings
    """

    cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
    dist_seq = cos(output_seq1, output_seq2)

    return dist_seq


if __name__ == '__main__':
    pass
