#!/bin/bash
#SBATCH --account=project_2000539
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=15
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=small
#SBATCH --gres=nvme:300

(
zcat ~/proj_finnlp_scratch/filip/pb_reparsed_cleaned_ner/delivery.*.conllu.gz | grep -P '^# text = ' | perl -pe 's/^# text = //' ;
zcat /scratch/project_2000539/pb_faiss/datain/para_sents.txt.gz; 
zcat /scratch/project_2000539/filip/tatoeba/eng-fin/train.trg.gz; 
cat /scratch/project_2000539/filip/tatoeba/eng-fin/test.trg) | python3 clean_data.py --outfile-pos /scratch/project_2000539/pb_faiss/all_data_pos.gz --outfile-neg /scratch/project_2000539/pb_faiss/all_data_neg.gz 

# all_data_pos.gz are all "positive" sentences, ie those which pass the filter
# all_data_neg.gz are all "negative" sentences, ie those dumped by the filter (this file is not used for anything after that)

export LC_ALL=C
export LC_COLLATE=C

zcat /scratch/project_2000539/pb_faiss/all_data_pos.gz | sort --parallel 15 -S 56G -T $LOCAL_SCRATCH | uniq | gzip > /scratch/project_2000539/pb_faiss/all_data_pos_uniq.gz

#all_data_pos_uniq.gz is the overall output here, used later on
