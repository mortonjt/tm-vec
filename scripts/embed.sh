#!/bin/bash

#SBATCH -N 1
#SBATCH --time=50:00:00
#SBATCH --job-name=embedding
#SBATCH --output=/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/embed.out
#SBATCH --error=/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/embed.err
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:32g
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=80g
#SBATCH --tmp 600000

export PATH="/cluster/apps/sfos/bin:/cluster/home/vbezshapkin/miniconda3/bin:$PATH"
source activate tmvec_slim
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
OUTPUT=$TMPDIR/embeddings.h5py

module load gcc/11.4.0
module load cuda/12.1.1
module load cudnn/8.9.2.26

python /nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/scripts/embed.py \
    --fasta-file /nfs/nas22/fs2202/biol_micro_sunagawa/Projects/STU/SSKIOTYTE_STU/scratch/raw/mmseqs/COG0048/COG0048_all.fasta \
    --model-local-path /nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/cache/models--facebook--esm2_t36_3B_UR50D/snapshots/476b639933c8baad5ad09a60ac1a87f987b656fc \
    --output-file $OUTPUT

mv $OUTPUT /nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/tmvec_data
