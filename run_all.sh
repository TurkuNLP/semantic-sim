for src in src trg
do
    for part in 00 01 02 03 04 05 06
    do
	echo "sbatch -J $src.$part -e eo/$src.$part.err -o eo/$src.$part.out distractors_sbatch.sh $src.$part"
    done
done
