#!/bin/bash
#SBATCH --account=Project_2005072
#SBATCH --time=00:15:00
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=6G
#SBATCH --gres=gpu:v100:1,nvme:5
#SBATCH --cpus-per-task=4


module load pytorch/1.11

whichmodel=sbert

DATADIR=/scratch/project_2005072/emil/faiss_distractors

zcat /scratch/project_2005072/emil/faiss_distractors/datain/para_sents.txt.gz | python3 print_nearest_from_faiss_index.py --sbert-model sentence-transformers/paraphrase-multilingual-mpnet-base-v2 --sbert-tokenizer sentence-transformers/paraphrase-multilingual-mpnet-base-v2 --fais /scratch/project_2005072/emil/faiss_distractors/faiss_index_filled_sbert.faiss --out /scratch/project_2005072/emil/faiss_distractors/nearest_sbert.ipkl

