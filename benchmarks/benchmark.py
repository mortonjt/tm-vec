import argparse
import os
import pickle
import time

import psutil

from tm_vec.embedding import ProtT5Encoder
from tm_vec.utils import generate_proteins

CACHE_DIR = "/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/cache"

# parse args
parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output", type=str)
parser.add_argument("--n_prots", type=int, default=200)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--backend", type=str, default="torch")
parser.add_argument("--compile", type=int, default=0)
parser.add_argument("--model_path", type=str)
parser.add_argument("--tokenizer_path", type=str)
args = parser.parse_args()


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


# Measure peak RAM
batch_ram_usages = []
batch_times = []
# only measure RAM for the current process
process = psutil.Process(os.getpid())
try:
    for i in range(0, len(proteins), BATCH_SIZE):
        peak_ram = 0
        batch_proteins = proteins[i:i + BATCH_SIZE]
        (inp, embeddings), timing = encode(batch_proteins, model)
        batch_times.append(timing)
        ram_usage = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        ram = max(peak_ram, ram_usage)
        batch_ram_usages.append(ram)
    benchmark_results["batch_times"] = batch_times
    benchmark_results["batch_ram_usages"] = batch_ram_usages
except Exception as e:
    print(f"Error benchmarkin: {e}")

# Store benchmark results in a pickle file
result_filename = f"benchmark_{args.batch_size}_{args.threads}_{args.backend}_{args.compile}.pkl"
with open(os.path.join(args.output, result_filename), "wb") as f:
    pickle.dump(benchmark_results, f)
