for fname in $(ls ~/proj_finnlp_scratch/pb_faiss/embedded_batches/unique_lines_part_00000000* | perl -pe 's/\_[0-9]+\.pt$//' | uniq)
do
    BASE=$(basename $fname)
    echo sbatch -o eo/qry_$BASE.out -e eo/qry_$BASE.err query_faiss_sbatch.sh ~/proj_finnlp_scratch/pb_faiss/all-by-all/$BASE.pickle ${fname}*.pt
done
