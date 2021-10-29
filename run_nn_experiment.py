import glob
import random
import gzip
import tqdm
import sys
import pickle
import numpy
import traceback
import itertools
import json
import re
import hashlib
import mmap_index
from sentence_transformers import SentenceTransformer

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

def remove_initial_dash(s): #removes "- "
    return re.sub("^\s*-+\s*","",s)

def s2hash(idx_sent):
    idx,sent=idx_sent
    return idx,hashlib.sha1(sent.encode("utf-8")).hexdigest()[:15]


if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument("--sentencefiles",nargs="+",help="Files with the sentences in the same order as in the index. Can be (and probably is) gz files")
    parser.add_argument("--sent-hash-index",help="hash2idx file")
    parser.add_argument("--qry-file",help="File of sentences used to build the NN pickle, these would be all para sentences")
    parser.add_argument("--paradata",help="para corpus test file")
    parser.add_argument("--knn-ipkl",help="the nearest neighbor files")
    parser.add_argument("--mmap-texts",help="memory-map text file index with sentence texts")
    args = parser.parse_args()

    if args.qry_file:
        #1) Load qry indices
        sent2qryidx={}
        f=open_possibly_gz_file(args.qry_file)
        for idx,line in enumerate(f):
            line=line.strip()
            line=remove_initial_dash(line)
            sent2qryidx[line]=idx

        #2) Load NN indices
        with open(args.sent_hash_index,"rb") as f:
            hash2idx=pickle.load(f)


        #1) Load the para data
        with open(args.paradata) as f:
            paradata=json.load(f)

            for e in paradata:
                t1=remove_initial_dash(e["txt1"])
                t2=remove_initial_dash(e["txt2"])
                label=e["label"]
                if t1 in sent2qryidx:
                    e["txt1_qryidx"]=sent2qryidx[t1]
                if t2 in sent2qryidx:
                    e["txt2_qryidx"]=sent2qryidx[t2]
                _,t1hash=s2hash((0,t1))
                _,t2hash=s2hash((0,t2))
                if t1hash in hash2idx:
                    e["txt1_nnidx"]=hash2idx[t1hash]
                if t2hash in hash2idx:
                    e["txt2_nnidx"]=hash2idx[t2hash]
        json.dump(paradata,sys.stdout,ensure_ascii=False,sort_keys=True,indent=2)
    elif args.knn_ipkl:
        all_nns=[]
        with open(args.knn_ipkl,"rb") as f:
            while True:
                try:
                    qry_ids,knn_ids=pickle.load(f)
                except EOFError:
                    break
                all_nns.extend(knn_ids.tolist())
        print("Got",len(all_nns))
        texts=mmap_index.Qry(args.mmap_texts)
        with open(args.paradata) as f:
            paradata=json.load(f)
            for e in paradata:
                if "txt1_qryidx" in e and "txt2_qryidx" in e and "txt1_nnidx" in e and "txt2_nnidx" in e:
                    nns=all_nns[e["txt1_qryidx"]]
                    try:
                        i=nns.index(e["txt2_nnidx"])
                    except ValueError:
                        i=None
                    print(e["label"],i,texts.get(e["txt1_nnidx"]),texts.get(e["txt2_nnidx"]),sep="\t")

                    nns=all_nns[e["txt2_qryidx"]]
                    try:
                        i=nns.index(e["txt1_nnidx"])
                    except ValueError:
                        i=None
                    print(e["label"]+"-rev",i,texts.get(e["txt2_nnidx"]),texts.get(e["txt1_nnidx"]),sep="\t")
                    
                # for q_id,knn_id in zip(qry_ids,knn_ids):
                #     print("**",qry_sents[int(q_id)])
                #     for nn_id in knn_id[:10]:
                #         print("  ",qry.get(nn_id))
                #     print("\n\n\n")

