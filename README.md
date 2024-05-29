# Paper
TM-Vec: template modeling vectors for fast homology detection and alignment: https://www.biorxiv.org/content/10.1101/2022.07.25.501437v1

[Embed sequences with TM-vec](https://colab.research.google.com/github/tymor22/tm-vec/blob/master/google_colabs/Embed_sequences_using_TM_Vec.ipynb)

# Installation

First create a conda environment with python=3.9 installed.  If you are using cpu, use

`conda create -n tmvec faiss-cpu python -c pytorch`

If the installation fails, you may need to install mkl via `conda install mkl=2021 mkl_fft `

Once your conda enviroment is installed and activated (i.e. `conda activate tmvec`), then install tmvec via
`pip install tmvec`. If you are using a GPU, you may need to reinstall the gpu version of pytorch.
See the [pytorch](https://pytorch.org/) webpage for more details.

# Run TM-Vec from the command line

To build database run the following command:
```
tmvec build-db \
    --input-fasta small_embed.fasta \
    --cache-dir cache \
    --tm-vec-config tm-vec/params.json \
    --tm-vec-model tm-vec/model.ckpt \
    --protrans-model cache/models--Rostlab--prot_t5_xl_uniref50 \
    --output db_test/small_fasta
```
To query a sequences against a database use:
```
tmvec search \
    --query /nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/small_embed.fasta \
    --database /nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/db_test/small_fasta.npz \
    --output db_test/result.tsv
```
