import numpy as np
from tqdm import tqdm

from tm_vec.embedding import ProtT5Encoder
from tm_vec.utils import generate_proteins

torch_model = ProtT5Encoder(
    "Rostlab/prot_t5_xl_uniref50",
    "Rostlab/prot_t5_xl_uniref50",
    cache_dir=
    "/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/cache",
    local_files_only=False)

onnx_model = ProtT5Encoder(
    "valentynbez/prot-t5-xl-uniref50-onnx",
    "Rostlab/prot_t5_xl_uniref50",
    cache_dir=
    "/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/cache",
    backend="onnx",
    local_files_only=False)

prots = generate_proteins(100)

for prot in tqdm(prots):
    _, torch_result = torch_model.batch_embed([prot])
    _, onnx_result = onnx_model.batch_embed([prot])
    assert np.allclose(torch_result, onnx_result, atol=1e-7)

print("All tests passed! ONNX is equal to Torch!")
