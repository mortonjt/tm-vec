#!/bin/bash

#SBATCH -N 1
#SBATCH --time=50:00:00
#SBATCH --job-name=embedding
#SBATCH --output=/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/embed.out
#SBATCH --error=/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/embed.err
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:24g
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8g
#SBATCH --tmp 10000

# onnx 1.6.0 bug https://github.com/danielgatis/rembg/issues/448
export OMP_NUM_THREADS=1

export PATH="/cluster/apps/sfos/bin:/cluster/home/vbezshapkin/miniconda3/bin:$PATH"
source activate tmvec_slim
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export OUTPUT=$TMPDIR/protT5_quantized_train_embeddings.h5py

module load gcc/11.4.0
module load cuda/12.1.1
module load cudnn/8.9.2.26

python /nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/scripts/bulk_embed.py \
    --fasta-file /nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/tmvec_data/train.fasta.gz \
    --model-path /nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/cache/models--Rostlab--prot_t5_xl_uniref50/snapshots/973be27c52ee6474de9c945952a8008aeb2a1a73 \
    --tokenizer-path /nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/cache/models--Rostlab--prot_t5_xl_uniref50/snapshots/973be27c52ee6474de9c945952a8008aeb2a1a73 \
    --max-tokens-per-batch 8096 \
    --output-file $OUTPUT

echo "Copying results to NFS"
rsync -a $OUTPUT /nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/tmvec_data/
echo "Done"
