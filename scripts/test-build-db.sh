#!/bin/bash

#SBATCH -N 1
#SBATCH --tmp=2000
#SBATCH --time=00:10:00
#SBATCH --job-name=prott5
#SBATCH --output=/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/test.log
#SBATCH --error=/nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/test.err
#SBATCH --gpus-per-node=1
#SBATCH --gres=gpumem:11g
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=2g

module load gcc/11.4.0
module load cuda/12.1.1
module load cudnn/8.9.2.26
export CUDA_HOME=$CUDA_BASE

export PATH="$PATH:/cluster/apps/sfos/bin:/cluster/home/vbezshapkin/miniconda3/bin"
source activate tmvec_slim

tmvec build-db \
    --input-fasta /nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/small_embed.fasta \
    --tm-vec-model /nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/tmvec_retrained/tmvec \
    --protrans-model /nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/cache/models--Rostlab--prot_t5_xl_uniref50/snapshots/973be27c52ee6474de9c945952a8008aeb2a1a73 \
    --cache-dir /nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/cache \
    --output /nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/db_test/small_fasta \
    --local

tmvec search \
    --input-fasta /nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/small_embed.fasta \
    --database /nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/db_test/cpu/small_fasta.npz \
    --output /nfs/cds-peta/exports/biol_micro_cds_gr_sunagawa/scratch/vbezshapkin/tm-vec/db_test/cpu/result.tsv
