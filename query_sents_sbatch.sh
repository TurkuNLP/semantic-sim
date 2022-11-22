#!/bin/bash
#SBATCH --account=Project_2002820
#SBATCH --time=6:15:00
##SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=6G
#SBATCH --gres=gpu:v100:1,nvme:60
#SBATCH --cpus-per-task=4


module load pytorch/1.11

whichmodel=sbert

DATADIR=/scratch/project_2000539/pb_faiss

zcat /scratch/project_2000539/pb_faiss/datain/para_sents.txt.gz | python3 print_nearest_from_faiss_index.py --bert-model TurkuNLP/bert-base-finnish-cased-v1 --fais ~/proj_finnlp_scratch/pb_faiss/faiss_index_filled_bert.faiss --out ~/proj_finnlp_scratch/pb_faiss/nearest_bert.ipkl

zcat /scratch/project_2000539/pb_faiss/datain/para_sents.txt.gz | python3 print_nearest_from_faiss_index.py --sbert-model /scratch/project_2000539/pb_faiss/sbert-cased-finnish-paraphrase --sbert-tokenizer TurkuNLP/bert-base-finnish-cased-v1 --fais ~/proj_finnlp_scratch/pb_faiss/faiss_index_filled_sbert.faiss --out ~/proj_finnlp_scratch/pb_faiss/nearest_sbert.ipkl

