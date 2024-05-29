import os

import faiss
import numpy as np


def save_database(headers: np.ndarray, embeddings: np.ndarray,
                  fasta_filepath: str, tm_vec_weights: str, tm_vec_conf: str,
                  protrans_model_path: str, output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    input_fasta = np.array(os.path.abspath(fasta_filepath))
    tm_vec_weights = np.array(os.path.abspath(tm_vec_weights))
    tm_vec_conf = np.array(os.path.abspath(tm_vec_conf))
    protrans_model_path = np.array(os.path.abspath(protrans_model_path))

    np.savez_compressed(output_path,
                        headers=headers,
                        embeddings=embeddings,
                        input_fasta=input_fasta,
                        tm_vec_weights=tm_vec_weights,
                        tm_vec_conf=tm_vec_conf,
                        protrans_model_path=protrans_model_path)


def build_vector_index(vector_db: np.ndarray):
    d = vector_db.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(vector_db)
    index.add(vector_db)
    return index


def load_database(path):
    database = np.load(path)
    input_fasta = database['input_fasta']
    tm_vec_weights = database['tm_vec_weights']
    tm_vec_conf = database['tm_vec_conf']
    protrans_model_path = database['protrans_model_path']
    headers = database['headers']
    embeddings = database['embeddings']

    # Build an indexed database
    index = build_vector_index(embeddings)

    return index, headers, input_fasta, tm_vec_weights, tm_vec_conf, protrans_model_path


def query(index, queries, k=10):
    faiss.normalize_L2(queries)
    values, indexes = index.search(queries, k)

    return values, indexes


def _format_id(ix, iy):
    """ Assumes that len(ix) > len(iy) """
    diff = len(ix) - len(iy)
    ix = ix + ' '
    iy = iy + ' ' * (diff + 1)
    return ix, iy


def format_ids(ix, iy):
    if len(ix) > len(iy):
        ix, iy = _format_id(ix, iy)
    else:
        iy, ix = _format_id(iy, ix)
    return ix, iy
