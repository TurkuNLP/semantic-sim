#!/bin/bash
#SBATCH --job-name=ecco_s2i
#SBATCH --account=project_2002820
#SBATCH --time=8:00:00
#SBATCH --mem=200000
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4  
#SBATCH --partition=small

module load pytorch/1.11
pip3 install -r requirements.txt
python3 index_sentences.py  --in-file /scratch/project_2002820/emil/ecco_faiss/ecco.jsonl --out-s2i /scratch/project_2002820/emil/ecco_faiss/all_data_pos_uniq_s2i.pkl
