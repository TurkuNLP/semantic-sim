#!/bin/bash
#SBATCH --account=Project_2002820
#SBATCH --time=2:15:00
##SBATCH --time=00:15:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1,nvme:60
#SBATCH --cpus-per-task=4


module load pytorch/1.9
echo "Installing faiss"
echo "Installing faiss" > /dev/stderr
pip3 install --user faiss-gpu
echo "Done installing faiss"
echo "Done installing faiss" > /dev/stderr

DATADIR=/users/ginter/proj_finnlp_scratch/pb_faiss
## PREPARE TINY SAMPLE ON WHICH THE INDEX IS TRAINED
python3 create_faiss_index.py --prepare-sample $DATADIR/faiss_index_train_sample.pt $DATADIR/embedded_batches/all_uniq_data_0_30.pkl #this is every 30th sentence in all of the data

## TRAIN THE INDEX ON THE 50% SAMPLE (of the 1/30 sample)
python3 create_faiss_index.py --train-faiss $DATADIR/faiss_index_pretrained.faiss $DATADIR/faiss_index_train_sample.pt

## FILL THE INDEX WITH ALL BATCHFILES
python3 create_faiss_index.py --fill-faiss $DATADIR/faiss_index_filled.faiss --pretrained-index $DATADIR/faiss_index_pretrained.faiss $DATADIR/embedded_batches/*.pkl

