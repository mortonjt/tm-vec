#!/usr/bin/env python3

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

from tm_vec.database import save_database
from tm_vec.utils import load_fasta_as_dict
from tm_vec.vectorizer import TMVec

logger = logging.getLogger(__name__)
handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '[%(asctime)s] %(module)s.%(funcName)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)

TMVEC_SEQ_LIM = 1024

parser = argparse.ArgumentParser(description='Process TM-Vec arguments',
                                 add_help=True)
parser.add_argument("--input-fasta",
                    type=Path,
                    required=True,
                    help=("Input proteins in fasta format to "
                          "construct the database. Can be gzipped."))
parser.add_argument("--tm-vec-model",
                    type=Path,
                    required=True,
                    help="Model path for TM-Vec embedding model")
parser.add_argument("--tm-vec-config",
                    type=Path,
                    required=True,
                    help=("Config path for TM-Vec embedding model. "
                          "This is used to encode the proteins as "
                          "vectors to construct the database."))
parser.add_argument(
    "--protrans-model",
    type=Path,
    default=None,
    required=False,
    help=("Model path for the ProtT5 embedding model. "
          "If this is not specified, then the model will "
          "automatically be downloaded to the cache directory."))

parser.add_argument("--cache-dir",
                    type=Path,
                    default=None,
                    required=False,
                    help=("Cache directory for the ProtT5 model."))

parser.add_argument("--output",
                    type=Path,
                    required=True,
                    help="Output path for the database files")

parser.add_argument("--local",
                    action="store_true",
                    help=("If this flag is set, then the model will only "
                          "use local files. This is useful for running the "
                          "script on a machine without internet access."))


def main(args):
    # Read in query sequences
    records = load_fasta_as_dict(args.input_fasta)

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
    tm_vec = TMVec(model_path=args.tm_vec_model,
                   config_path=args.tm_vec_config,
                   cache_dir=args.cache_dir,
                   protlm_path=args.protrans_model,
                   protlm_tokenizer_path=args.protrans_model,
                   local_files_only=args.local)
    # Embed all query sequences
    encoded_database = tm_vec.vectorize_proteins(seqs[:2])

    # Save database
    metdata = np.array(headers)
    save_database(metdata, encoded_database, args.input_fasta,
                  args.tm_vec_model, args.tm_vec_config, args.protrans_model,
                  args.output)
    logger.info(
        "Please, do not move or rename input FASTA file, TM-Vec model and config files and "
        "ProtT5 model. They will be used to for sequence search. It is a design feature to "
        "ensure the consistency of the models used. If the location has chaged "
        "you can manually modify the database.")


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
