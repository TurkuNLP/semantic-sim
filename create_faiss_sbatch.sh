#!/bin/bash
#SBATCH --account=Project_2000539
#SBATCH --time=05:15:00
##SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1,nvme:60
#SBATCH --cpus-per-task=4


module load python-data/3.7.6-1
source /projappl/project_2000539/faiss_distractors/venv-faissgpu/bin/activate

#src or trg
WHICH=$1 #src or trg

## PREPARE 5% SAMPLE ON WHICH THE INDEX IS TRAINED
#python3 create_faiss_index.py --prepare-sample /scratch/project_2000539/filip/tatoeba/eng-fin/faiss_sample_train_$WHICH.pt /scratch/project_2000539/filip/tatoeba/eng-fin/embedded_${WHICH}.*/*.pt

## TRAIN THE INDEX ON THE 5% SAMPLE
#python3 create_faiss_index.py --train-faiss /scratch/project_2000539/filip/tatoeba/eng-fin/trainedindex_train_$WHICH.faiss /scratch/project_2000539/filip/tatoeba/eng-fin/faiss_sample_train_$WHICH.pt

## FILL THE INDEX WITH ALL BATCHFILES
## This relies on the fact that the .*/*.pt can be sorted, so the vectors are stored
## in the correct order, or else everything gets messed up :|
python3 create_faiss_index.py --fill-faiss /scratch/project_2000539/filip/tatoeba/eng-fin/filled_index_train_$WHICH.faiss --pretrained-index /scratch/project_2000539/filip/tatoeba/eng-fin/trainedindex_train_$WHICH.faiss /scratch/project_2000539/filip/tatoeba/eng-fin/embedded_${WHICH}.*/*.pt

