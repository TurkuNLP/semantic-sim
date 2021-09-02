for part in $(seq 0 29)
do
    echo "sbatch -e eo/embed_$part-of-30.err -o eo/embed_$part-of-30.out -J e$part embed_sbatch.sh $part 30 ~/proj_finnlp_scratch/pb_faiss/all_uniq_data.uniq.gz ~/proj_finnlp_scratch/pb_faiss/embedded_batches/all_uniq_data_${part}_30.pkl"
done

