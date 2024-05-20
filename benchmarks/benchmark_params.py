import os

import numpy as np
from tqdm import tqdm

from tm_vec.embedding import ProtT5Encoder
from tm_vec.utils import generate_proteins

output_folder = "benchmark_data"
tokenizer = "rostlab/prot_t5_xl_uniref50"
current_folder = os.getcwd()

# parameter matrix
BATCH_SIZES = [1, 2, 4, 8, 16, 32]
NUM_THREADS = [1, 2, 4, 8, 16]
# Model configurations
model_configs = [
    {
        "path": "rostlab/prot_t5_xl_uniref50",
        "backend": "torch",
        "compile_model": 0
    },
    {
        "path": "rostlab/prot_t5_xl_uniref50",
        "backend": "torch",
        "compile_model": 1
    },
    {
        "path":
        os.path.join(current_folder,
                     "../onnx/prot_t5_xl_uniref50_onnx_optimized"),
        "backend":
        "onnx",
        "compile_model":
        0
    },
]

########### TEST FOR CORRECT OUTPUTS ######################

torch_model = ProtT5Encoder(
    "Rostlab/prot_t5_xl_uniref50",
    "Rostlab/prot_t5_xl_uniref50",
    cache_dir=
    "/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/cache",
    local_files_only=False)

onnx_model = ProtT5Encoder(
    os.path.join(current_folder, "../onnx/prot_t5_xl_uniref50_onnx_optimized"),
    "Rostlab/prot_t5_xl_uniref50",
    cache_dir=
    "/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/cache",
    backend="onnx",
    local_files_only=False)

prots = generate_proteins(10)

for prot in tqdm(prots):
    _, torch_result = torch_model.batch_embed([prot])
    _, onnx_result = onnx_model.batch_embed([prot])
    try:
        assert np.allclose(torch_result, onnx_result, atol=1e-3)
    except AssertionError:
        print("ONNX is not equal to Torch!")

print("All tests passed! ONNX is equal to Torch!")

####################################################################

for batch in tqdm(BATCH_SIZES):
    n_prots = batch * 5
    for model_conf in model_configs:
        # run script
        os.system(f"python benchmark.py --n_prots {n_prots} "
                  f"--batch_size {batch} "
                  f"--threads 1 --backend {model_conf['backend']} "
                  f"--compile {model_conf['compile_model']} "
                  f"--model_path {model_conf['path']} "
                  f"--tokenizer_path {tokenizer} "
                  f"--output {output_folder}")

# test thread scaling ONNX
batch = 32
n_prots = batch * 5
model_conf = model_configs[2]
for thread in NUM_THREADS:
    os.system(f"python benchmark.py --n_prots {n_prots} "
              f"--batch_size {batch} "
              f"--threads {thread} --backend {model_conf['backend']} "
              f"--compile {model_conf['compile_model']} "
              f"--model_path {model_conf['path']} "
              f"--tokenizer_path {tokenizer} "
              f"--output {output_folder}")
