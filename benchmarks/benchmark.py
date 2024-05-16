import argparse
import pickle
import time

import numpy as np

from tm_vec.embedding import ProtT5Encoder

PROTEIN_ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
CACHE_DIR = "/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/cache"

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("--n_prots", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--backend", type=str, default="torch")
parser.add_argument("--compile", type=int, default=0)
parser.add_argument("--model_path", type=str)
parser.add_argument("--tokenizer_path", type=str)
args = parser.parse_args()


# Generate random proteins
def generate_proteins(n_prots):
    np.random.seed(42)
    proteins = []
    for _ in range(n_prots):
        prot = "".join(
            np.random.choice(list(PROTEIN_ALPHABET),
                             size=np.random.randint(20, 100)))
        proteins.append(prot)
    return proteins


# Timeit wrapper
def timeit(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter() - start
        return result, end

    return wrapper


# Generate proteins
proteins = generate_proteins(n_prots=args.n_prots)

# Benchmark models
benchmark_results = {}
BATCH_SIZE = args.batch_size
# save params to dictionary
benchmark_results["params"] = vars(args)

# Load the model
start = time.time()
model = ProtT5Encoder(args.model_path,
                      args.tokenizer_path,
                      CACHE_DIR,
                      backend=args.backend,
                      compile_model=args.compile,
                      threads=args.threads)
end = time.time()
benchmark_results["model_load_time"] = end - start


# Time the model
@timeit
def encode(proteins, model):
    return model.batch_embed(proteins)


try:
    batch_times = []
    for i in range(0, len(proteins), BATCH_SIZE):
        batch_proteins = proteins[i:i + BATCH_SIZE]
        (inp, embeddings), timing = encode(batch_proteins, model)
        batch_times.append(timing)
    benchmark_results["batch_times"] = batch_times
except Exception as e:
    print(f"Error benchmarkin: {e}")

# Store benchmark results in a pickle file
with open(
        f"benchmark_{args.batch_size}_{args.threads}_{args.backend}_{args.compile}.pkl",
        "wb") as f:
    pickle.dump(benchmark_results, f)
