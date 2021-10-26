import gzip
import pickle
import json
import sys
import tqdm

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
    parser.add_argument("--sentencefiles",nargs="+",help="Files with the sentences in the same order as in the index. Can be (and probably is) gz files and must match the knn_ids")
    parser.add_argument("--knn-ipkl",help="The ipkl file with the knn data")
    args = parser.parse_args()

    #Read query texts first, we will need these to be able to print the outcome
    qry_sents=[]
    for fname in tqdm.tqdm(args.qry_sentences):
        f=open_possibly_gz_file(fname)
        qry_sents.extend((l.strip() for l in f if l.strip()))
        f.close()
    print(f"Read {len(qry_sents)} q-sentences",file=sys.stderr,flush=True)

    #Read all texts first, we will need these to be able to print the outcome
    all_sents=[]
    for fname in tqdm.tqdm(args.sentencefiles):
        f=open_possibly_gz_file(fname)
        all_sents.extend((l.strip() for l in f if l.strip()))
        f.close()
    print(f"Read {len(all_sents)} sentences",file=sys.stderr,flush=True)

    print(qry_sents[:5])
    print()
    print(all_sents[:5])

    all_sents=set(all_sents)
    found=0
    for x in qry_sents:
        if x in all_sents:
            found+=1
    print(f"Found {found} / {len(qry_sents)}")
    sys.exit()

    with open(args.knn_ipkl,"rb") as f:
        while True:
            try:
                qry_ids,knn_ids=pickle.load(f)
            except EOFError:
                break
            for q_id,knn_id in zip(qry_ids,knn_ids):
                print("**",qry_sents[int(q_id)])
                for nn_id in knn_id[:5]:
                    print("  ",all_sents[nn_id])
                print("\n\n\n")

