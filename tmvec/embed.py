import argparse
import logging
import sys

import h5py
from tqdm import tqdm

from tmvec.embedding import ProtT5Encoder
from tmvec.utils import create_batched_sequence_datasest, load_fasta_as_dict

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '[%(asctime)s] %(module)s.%(funcName)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

# embedder = ESMEncoder(
#     model_path=args.model_local_path,
#     cache_dir=
#     "/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/cache",
#     device=device,
#     local_files_only=True,
#     pooling_layer=False)

# create arguments
parser = argparse.ArgumentParser(description="Embed sequences using ProtLM.")
parser.add_argument("--fasta-file",
                    type=str,
                    required=True,
                    help="Path to input FASTA file.")
parser.add_argument("--model-path",
                    type=str,
                    required=True,
                    help="Path to local model.")
parser.add_argument("--tokenizer-path",
                    type=str,
                    required=True,
                    help="Path to tokenizer.")
parser.add_argument("--output-file",
                    type=str,
                    required=True,
                    help="Path to output HDF5 file.")
parser.add_argument("--max-tokens-per-batch",
                    type=int,
                    default=1024,
                    help="Maximum tokens per batch.")
args = parser.parse_args()


def main(args):
    # load sequences
    seqs_dict = load_fasta_as_dict(args.fasta_file)
    # filter sequences longer than 1024
    seqs_dict = {k: v for k, v in seqs_dict.items() if len(v) <= 1024}

    # prepare batches
    sequences = create_batched_sequence_datasest(
        seqs_dict, max_tokens_per_batch=args.max_tokens_per_batch)
    embedder = ProtT5Encoder(
        args.model_path,
        args.tokenizer_path,
        cache_dir=
        "/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/cache",
        backend="torch",
        compile_model=True,
        local_files_only=True)

    with h5py.File(args.output_file, "w") as f:
        f.attrs["model_path"] = embedder.model_path
        with tqdm(total=len(seqs_dict)) as pbar:
            for batch_headers, batch_sequences in sequences:
                try:
                    emb = embedder.get_sequence_embeddings(batch_sequences)

                except RuntimeError as e:
                    if e.args[0].startswith("CUDA out of memory"):
                        if len(batch_sequences) > 1:
                            logger.info(
                                "Failed (CUDA out of memory) to predict "
                                f"batch of size {len(batch_sequences)}. "
                                "Try lowering `--max-tokens-per-batch`.")
                        else:
                            logger.info(
                                "Failed (CUDA out of memory) on sequence "
                                f"of length {len(batch_sequences[0])}.")
                        continue

                pbar.update(len(batch_sequences))
                # save embeddings
                for header, sequence, embedding in zip(batch_headers,
                                                       batch_sequences, emb):
                    grp = f.create_group(header)
                    grp.create_dataset("emb", data=embedding)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
