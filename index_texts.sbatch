#!/bin/bash
#SBATCH --job-name=idxtxt
#SBATCH --account=project_2000539
#SBATCH --time=6:0:0
#SBATCH --mem=140000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  
#SBATCH --partition=small

# Add module load commands here


# Add commands to be executed here
zcat ~/proj_finnlp_scratch/pb_faiss/all_data_pos_uniq.gz | python3 index_sentences.py --out-s2i ~/proj_finnlp_scratch/pb_faiss/all_data_pos_uniq_s2i.pkl
