import logging
import sys
from pathlib import Path

import click
from click._compat import get_text_stderr
from click.exceptions import UsageError
from click.utils import echo

from tmvec import TMVEC_SEQ_LIM, __version__
from tmvec.database import (get_metadata_for_neighbors, load_database, query,
                            save_database)
from tmvec.utils import load_fasta_as_dict, save_embeddings, save_results
from tmvec.vectorizer import TMVec

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '[%(asctime)s] %(module)s.%(funcName)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def _show_usage_error(self, file=None):
    if file is None:
        file = get_text_stderr()
    color = None
    if self.ctx is not None:
        color = self.ctx.color
        echo(self.ctx.get_help() + '\n', file=file, color=color)
    echo('Error: %s' % self.format_message(), file=file, color=color)


UsageError.show = _show_usage_error


@click.group()
@click.version_option(__version__, "-v", "--version", prog_name="tmvec")
def main():
    "tmvec: A tool for quick identification of remote homologues via protein embeddings"


@main.command()
@click.option("--input-fasta",
              type=Path,
              required=True,
              help=("Input proteins in fasta format to "
                    "construct the database. Can be gzipped."))
@click.option("--output",
              type=Path,
              required=True,
              help="Output path for the database files.")
@click.option(
    "--tm-vec-model",
    type=Path,
    default=None,
    required=False,
    help="Path for TM-Vec model folder from HuggingFace. "
    "Only required if model is supposed to be run on a computer without "
    "internet access.")
@click.option(
    "--protrans-model",
    type=Path,
    default=None,
    required=False,
    help=("Model path for the ProtT5 embedding model. "
          "If this is not specified, then the model will "
          "automatically be downloaded to the cache directory. "
          "Only required if model is supposed to be run on a computer without "
          "internet access."))
@click.option(
    "--cache-dir",
    type=Path,
    default=None,
    required=False,
    help=("Cache directory for the ProtT5 model. Might be useful if temporary "
          "directory runs out of space."))
@click.option("--local",
              is_flag=True,
              default=False,
              help=("If this flag is set, then the model will only "
                    "use local files. This is useful for running the "
                    "script on a machine without internet access."))
def build_db(input_fasta, output, tm_vec_model, protrans_model, cache_dir,
             local):
    """
    Build a database of protein vectors for fast structure search.
    """

    # Read in query sequences
    records = load_fasta_as_dict(input_fasta)

    # sort sequences by length
    prefilter = len(records)
    # remove sequences longer than 1024
    records = {k: v for k, v in records.items() if len(v) <= TMVEC_SEQ_LIM}
    postfilter = len(records)
    logger.info(
        f"Removed {prefilter - postfilter} sequences longer than {TMVEC_SEQ_LIM} residues."
    )

    headers, seqs = zip(*records.items())
    # Load model
    tm_vec = TMVec(model_folder=tm_vec_model,
                   cache_dir=cache_dir,
                   protlm_path=protrans_model,
                   protlm_tokenizer_path=protrans_model,
                   local_files_only=local)
    # Embed all query sequences
    encoded_database = tm_vec.vectorize_proteins(seqs)

    # Save database
    save_database(headers, encoded_database, input_fasta, tm_vec_model,
                  protrans_model, output)

    logger.info(
        "Please, do not move or rename input FASTA file, TM-Vec model and config files and "
        "ProtT5 model. They will be used to for sequence search. It is a design feature to "
        "ensure the consistency of the models used. If the location has chaged "
        "you can manually modify the database.")


@main.command()
@click.option("--input-fasta",
              type=Path,
              required=True,
              help="Path to input FASTA file. Can be a gzipped.")
@click.option("--database",
              type=Path,
              required=True,
              help="Path to the TM-Vec database file.")
@click.option("--output",
              type=Path,
              required=True,
              help="Output path for the results.")
@click.option("--output-format",
              type=click.Choice(["tabular"]),
              default="tabular",
              help="Output format for the results.")
@click.option("--output-embeddings",
              type=Path,
              default=None,
              help="Output path for the query embeddings.")
@click.option("--k-nearest",
              type=int,
              default=5,
              help="Number of nearest neighbors to return.")
@click.option("--deepblast-model",
              type=Path,
              default=None,
              help="Path to the DeepBLAST model.")
@click.option("--local",
              is_flag=True,
              default=False,
              help=("If this flag is set, then the model will only "
                    "use local files. This is useful for running the "
                    "script on a machine without internet access."))
def search(input_fasta, database, output, output_format, output_embeddings,
           k_nearest, deepblast_model, local):
    """
    Search for similar proteins in a database using TM-Vec embeddings and align them with DeepBLAST.
    """

    # Read in query sequences
    records = load_fasta_as_dict(input_fasta)

    prefilter = len(records)
    # remove sequences longer than 1024
    records = {k: v for k, v in records.items() if len(v) <= TMVEC_SEQ_LIM}

    postfilter = len(records)
    headers, seqs = zip(*records.items())
    logger.info(
        f"Removed {prefilter - postfilter} sequences longer than {TMVEC_SEQ_LIM} residues."
    )

    # Load database
    query_database, index, target_headers, \
        input_fasta, tm_vec_model, protrans_model = load_database(database)

    # Load model
    tm_vec = TMVec(model_folder=tm_vec_model,
                   cache_dir=None,
                   protlm_path=protrans_model,
                   protlm_tokenizer_path=protrans_model,
                   local_files_only=local)

    # Embed all query sequences
    queries = tm_vec.vectorize_proteins(seqs)

    # Return the nearest neighbors
    values, indexes = query(index, queries, k_nearest)
    # Return the metadata for the nearest neighbor results
    near_ids = get_metadata_for_neighbors(indexes, target_headers)

    # tm_vec.delete_vectorizer()  # need to clear space for DeepBLAST aligner

    # # Alignment section
    # # If we are also aligning proteins, load tm_vec align and
    # # align nearest neighbors if true
    # if args.deepblast_model is not None and args.output_format == 'alignment':
    #     # moving imports here because it may take a very long time
    #     from deepblast.dataset.utils import states2alignment
    #     from deepblast.utils import load_model

    #     align_model = load_model(
    #         args.deepblast_model, lm=model, tokenizer=tokenizer,
    #         alignment_mode=args.alignment_mode,
    #         device=device)

    #     align_model = align_model.to(device)
    #     seq_db = FastaFile(args.database_fasta)

    #     alignments = []
    #     with open(args.output, 'w') as fh:
    #         for i in range(I.shape[0]):
    #             for j in range(I.shape[1]):
    #                 x = flat_seqs[i]
    #                 seq_i = metadata_database[I[i, j]]
    #                 y = seq_db.fetch(seq_i)
    #                 try:
    #                     pred_alignment = align_model.align(x, y)
    #                     # Note : there is an edge case that smith-waterman will throw errors,
    #                     # but needleman-wunsch won't.
    #                     x_aligned, y_aligned = states2alignment(pred_alignment, x, y)
    #                     alignments_i = [x_aligned, pred_alignment, y_aligned]
    #                     # sWrite out the alignments
    #                     x, s, y = alignments_i
    #                     # TODO : not clear how to get the sequence IDS
    #                     ix, iy = format_ids(headers[i], seq_i)
    #                     fh.write(ix + ' ' + x + '\n')
    #                     fh.write(iy + ' ' + y + '\n\n')
    #                 except:
    #                     print(f'No valid alignments found for {headers[i]} {seq_i}')

    save_results(values, near_ids, target_headers, output_format, output)
    save_embeddings(queries, output_embeddings)


if __name__ == '__main__':
    main()
