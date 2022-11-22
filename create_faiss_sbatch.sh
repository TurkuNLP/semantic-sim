#!/bin/bash
#SBATCH --account=Project_2002820
#SBATCH --time=30:00:00
##SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=24G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1,nvme:60
#SBATCH --cpus-per-task=4


module load pytorch/1.11
echo "Installing faiss"
echo "Installing faiss" > /dev/stderr
pip3 install --user faiss-gpu
echo "Done installing faiss"
echo "Done installing faiss" > /dev/stderr

model=sbert
DATADIR=/scratch/project_2005072/emil/faiss_distractors
## PREPARE TINY SAMPLE ON WHICH THE INDEX IS TRAINED
python3 create_faiss_index.py --prepare-sample $DATADIR/faiss_index_train_sample_${model}.pt $DATADIR/embedded_batches_${model}/all_uniq_data_0_50.pkl #this is every 50th sentence in all of the data

## TRAIN THE INDEX ON THE 50% SAMPLE (of the 1/50 sample)
python3 create_faiss_index.py --train-faiss $DATADIR/faiss_index_pretrained_${model}.faiss $DATADIR/faiss_index_train_sample_${model}.pt

## FILL THE INDEX WITH ALL BATCHFILES
python3 create_faiss_index.py --fill-faiss $DATADIR/faiss_index_filled_${model}.faiss --pretrained-index $DATADIR/faiss_index_pretrained_${model}.faiss $DATADIR/embedded_batches_${model}/*.pkl

