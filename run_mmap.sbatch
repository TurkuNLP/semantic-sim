#!/bin/bash
#SBATCH --job-name=ecco_mmap
#SBATCH --account=project_2005072
#SBATCH --time=10:00:00
#SBATCH --mem=200000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  
#SBATCH --partition=small

module load pytorch/1.11
pip3 install -r requirements.txt
zcat /scratch/project_2002820/emil/ecco_faiss/ecco.jsonl.gz | python3  mmap_index.py --index-lines-to all_data_pos_uniq
