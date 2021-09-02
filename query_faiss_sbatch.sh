#!/bin/bash
#SBATCH --account=Project_2002820
#SBATCH --time=6:15:00
##SBATCH --time=00:15:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=6G
#SBATCH --gres=gpu:v100:1,nvme:60
#SBATCH --cpus-per-task=4


module load python-data/3.7.6-1
source /projappl/project_2000539/faiss_distractors/venv-faissgpu/bin/activate

#src or trg
#WHICH=$1 #src or trg

DATADIR=/scratch/project_2000539/pb_faiss
OUT=$1
shift

python3 faiss_query_all_by_all.py --prefilled-index $DATADIR/faiss_index_filled.faiss --save $OUT $*

#python3 print_nearest.py --sentencefile /scratch/project_2000539/filip/tatoeba/eng-fin/train_uniq.$WHICH.gz --nnfile /scratch/project_2000539/filip/tatoeba/eng-fin/all-by-all_train_$WHICH.pt --outfile /scratch/project_2000539/filip/tatoeba/eng-fin/train_uniq.distractors.$WHICH.gz
