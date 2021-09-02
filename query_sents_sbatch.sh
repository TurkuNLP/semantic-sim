#!/bin/bash
#SBATCH --account=Project_2002820
##SBATCH --time=6:15:00
#SBATCH --time=00:15:00
#SBATCH --partition=gputest
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=6G
#SBATCH --gres=gpu:v100:1,nvme:60
#SBATCH --cpus-per-task=4


module load pytorch/1.9

DATADIR=/scratch/project_2000539/pb_faiss
zcat $DATADIR/pbank_nn.gz | python3 faiss_query_sents.py --prefilled-index $DATADIR/faiss_index_filled.faiss

#python3 print_nearest.py --sentencefile /scratch/project_2000539/filip/tatoeba/eng-fin/train_uniq.$WHICH.gz --nnfile /scratch/project_2000539/filip/tatoeba/eng-fin/all-by-all_train_$WHICH.pt --outfile /scratch/project_2000539/filip/tatoeba/eng-fin/train_uniq.distractors.$WHICH.gz
