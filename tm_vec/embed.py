import argparse
import logging

import h5py
import torch
from tqdm import tqdm

from tm_vec.embedding import ESMEncoder
from tm_vec.utils import create_batched_sequence_datasest, load_fasta_as_dict

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# create arguments
parser = argparse.ArgumentParser(description="Embed sequences using ESM")
parser.add_argument("--fasta-file",
                    type=str,
                    required=True,
                    help="Path to input FASTA file.")
parser.add_argument("--model-local-path",
                    type=str,
                    required=True,
                    help="Path to local model.")
parser.add_argument("--output-file",
                    type=str,
                    required=True,
                    help="Path to output HDF5 file.")
parser.add_argument("--max-tokens-per-batch",
                    type=int,
                    default=1024,
                    help="Maximum tokens per batch.")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load sequences
seqs_dict = load_fasta_as_dict(args.fasta_file)
# # filter sequences longer than 1024
# seqs_dict = {k: v for k, v in seqs_dict.items() if len(v) <= 1024}
# sort sequences by length
seqs_dict = dict(sorted(seqs_dict.items(), key=lambda x: len(x[1])))
# prepare batches
sequences = create_batched_sequence_datasest(seqs_dict,
                                             max_tokens_per_batch=2048)

# load embedding model
logger.info(f"Loaded {len(seqs_dict)} unique sequences.")

embedder = ESMEncoder(
    model_path=args.model_local_path,
    cache_dir=
    "/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/cache",
    device=device,
    local_files_only=True,
    pooling_layer=False)

logger.info("Loaded embedder.")

with h5py.File(args.output_file, "w") as f:
    f.attrs["model_path"] = embedder.model_path
    for batch_headers, batch_sequences in tqdm(sequences):
        try:
            emb = embedder.batch_embed(batch_sequences)

        except RuntimeError as e:
            if e.args[0].startswith("CUDA out of memory"):
                if len(batch_sequences) > 1:
                    logger.info("Failed (CUDA out of memory) to predict "
                                f"batch of size {len(batch_sequences)}. "
                                "Try lowering `--max-tokens-per-batch`.")
                else:
                    logger.info("Failed (CUDA out of memory) on sequence "
                                f"of length {len(batch_sequences[0])}.")
                continue

        logger.info(f"Predicted embeddings for batch of size {len(emb)}.")
        for header, sequence, embedding in zip(batch_headers, batch_sequences,
                                               emb):
            logger.info(f"Writing embedding for sequence {header}.")
            grp = f.create_group(header)
            grp.create_dataset("seq", data=sequence)
            grp.create_dataset("emb", data=embedding)
