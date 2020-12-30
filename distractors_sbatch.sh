#!/bin/bash
#SBATCH --account=Project_2000539
#SBATCH --time=09:15:00
##SBATCH --time=47:56:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=8G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1,nvme:60
#SBATCH --cpus-per-task=4






module load python-data/3.7.6-1
source /projappl/project_2000539/faiss_distractors/venv-faissgpu/bin/activate

#src or trg
WHICH=$1 #src.01 etc

OUTDIR=/scratch/project_2000539/filip/tatoeba/eng-fin/embedded_${WHICH}
mkdir -p $OUTDIR
cat /scratch/project_2000539/filip/tatoeba/eng-fin/train_part.$WHICH | python3 embed.py --bert-model /projappl/project_2000539/faiss_distractors/bi-bert-80k-transformers --out $OUTDIR/batches


