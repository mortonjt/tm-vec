from tm_vec.embedding import ESMEncoder
import torch
from Bio import SeqIO

import numpy as np
import pandas as pd
from tqdm import tqdm
import gzip
from typing import Dict, Generator, Tuple, List

import logging
import h5py
import os
import tempfile
from typing import Dict

tmpfolder = tempfile.gettempdir()

def load_fasta_as_dict(fasta_file: str, max_seqs = None) -> Dict[str, str]:
    """
    Load FASTA file as dict of headers to sequences

    Args:
        fasta_file (str): Path to FASTA file. Can be compressed.

    Returns:sq
        Dict[str, str]: Dictionary of FASTA entries sorted by length.
    """

    seqs_dict = {}
    with gzip.open(fasta_file, "rt") as f:
        for i, record in enumerate(SeqIO.parse(f, "fasta")):
            seqs_dict[record.id] = str(record.seq)
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


logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

fasta_file = "/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/tmvec_data/train.fasta.gz"
model_local_path = "/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/cache/models--facebook--esm2_t36_3B_UR50D/snapshots/476b639933c8baad5ad09a60ac1a87f987b656fc"
output_file = tmpfolder + "/train_embeddings.h5py"

seqs_dict = load_fasta_as_dict(fasta_file)
# filter sequences longer than 1024
seqs_dict = {k: v for k, v in seqs_dict.items() if len(v) <= 1024}
# sort sequences by length
seqs_dict = dict(sorted(seqs_dict.items(), key=lambda x: len(x[1])))
# prepare batches
sequences = create_batched_sequence_datasest(seqs_dict, max_tokens_per_batch=3072)


# load embedding model
logger.info(f"Loaded {len(seqs_dict)} unique sequences.")

embedder = ESMEncoder(model_path=model_local_path,
                      cache_dir="/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/cache",
                      device=device,
                      local_files_only = True,
                      pooling_layer = False)

logger.info("Loaded embedder.")

with h5py.File(output_file, "w") as f:
    f.attrs["model_path"] = embedder.model_path
    for batch_headers, batch_sequences in tqdm(sequences):
        try:
            emb = embedder.batch_embed(batch_sequences)

        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                if len(batch_sequences) > 1:
                    logger.info(
                        f"Failed (CUDA out of memory) to predict batch of size {len(batch)}. "
                          "Try lowering `--max-tokens-per-batch`."
                    )
                else:
                    logger.info(
                        f"Failed (CUDA out of memory) on sequence of length {len(batch[0])}."
                    )
                continue

        logger.info(f"Predicted embeddings for batch of size {len(batch_headers)}.")
        for header, sequence, embedding in zip(batch_headers, batch_sequences, emb):
            logger.info(f"Writing embedding for sequence {header}.")
            grp = f.create_group(header)
            grp.create_dataset("seq", data=sequence)
            grp.create_dataset("emb", data=embedding)







