import gzip
import pickle
import json
import sys
import tqdm
import mmap_index

def open_possibly_gz_file(f_or_name):
    if f_or_name is None:
        return None
    elif isinstance(f_or_name,str): #a file name of some sort
        if f_or_name.endswith(".gz"):
            return gzip.open(f_or_name,"rt")
        elif os.path.exists(f_or_name):
            return open(f_or_name)
        elif os.path.exists(f_or_name+".gz"):
            return gzip.open(f_or_name+".gz","rt")
        else:
            raise ValueError(f"No such file {f_or_name}, neither plain nor zipped")
    else:
        return f_or_name


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--qry-sentences",nargs="+",help="File(s) with the sentences in the same order in which they were queried in print_nearest_...")
    parser.add_argument("--sentenceindex",help="File prefix with sentence mmappable index produced by mmap_index.py")
    parser.add_argument("--knn-ipkl",help="The ipkl file with the knn data")
    args = parser.parse_args()

    #Read query texts first, we will need these to be able to print the outcome
    qry_sents=[]
    for fname in tqdm.tqdm(args.qry_sentences):
        f=open_possibly_gz_file(fname)
        qry_sents.extend((l.strip() for l in f if l.strip()))
        f.close()
    print(f"Read {len(qry_sents)} q-sentences",file=sys.stderr,flush=True)

    qry=mmap_index.Qry(args.sentenceindex)


    with open(args.knn_ipkl,"rb") as f:
        while True:
            try:
                qry_ids,knn_ids=pickle.load(f)
            except EOFError:
                break
            for q_id,knn_id in zip(qry_ids,knn_ids):
                print("**",qry_sents[int(q_id)])
                for nn_id in knn_id[:10]:
                    print("  ",qry.get(nn_id))
                print("\n\n\n")

