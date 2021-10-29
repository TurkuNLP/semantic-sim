#python3 run_nn_experiment.py --sent-h ~/proj_finnlp_scratch/pb_faiss/all_data_pos_uniq_s2i.pkl --qry ~/proj_finnlp_scratch/pb_faiss/datain/para_sents.txt.gz --para ../Turku-paraphrase-corpus/data-fi/test.json
python3 run_nn_experiment.py --knn ~/proj_finnlp_scratch/pb_faiss/nearest_bert.ipkl --para test.json --mmap ~/proj_finnlp_scratch/pb_faiss/all_data_pos_uniq > res_nearest.bert
python3 run_nn_experiment.py --knn ~/proj_finnlp_scratch/pb_faiss/nearest_sbert.ipkl --para test.json --mmap ~/proj_finnlp_scratch/pb_faiss/all_data_pos_uniq > res_nearest.sbert
