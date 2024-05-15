import numpy as np
import faiss


def load_database(path):
    lookup_database = np.load(path)
    #Build an indexed database
    d = lookup_database.shape[1]
    index = faiss.IndexFlatIP(d)
    faiss.normalize_L2(lookup_database)
    index.add(lookup_database)

    return (index)


def query(index, queries, k=10):
    faiss.normalize_L2(queries)
    D, I = index.search(queries, k)

    return (D, I)


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