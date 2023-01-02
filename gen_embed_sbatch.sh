for part in $(seq 0 49)
do
    echo "sbatch -e eo2/sbert_embed_$part-of-50.err -o eo2/sbert_embed_$part-of-50.out -J e$part embed_sbatch.sh $part 50 /scratch/project_2002820/emil/ecco_faiss/ecco.jsonl /scratch/project_2002820/emil/faiss_distractors/embedded_batches_sbert/all_uniq_data_${part}_50.pkl"
    #echo "sbatch -e eo2/bert_embed_$part-of-30.err -o eo2/bert_embed_$part-of-30.out -J eb$part embed_sbatch.sh $part 30 ~/proj_finnlp_scratch/pb_faiss/all_data_pos_uniq.gz ~/proj_finnlp_scratch/pb_faiss/embedded_batches_bert/all_uniq_data_${part}_30.pkl"
done

