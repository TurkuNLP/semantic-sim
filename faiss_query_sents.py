import faiss
import torch
import glob
import random
import tqdm
import pickle
import numpy
import sys
import gzip
import embed_data
import embed
import transformers
import time

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries",default=None,help="The sentences to query with one per line")
    parser.add_argument("--indexed-sentences",default=None,help="List of the sentences for which the index was built")
    parser.add_argument("--prefilled-index",default=None,help="Name of the prefilled index to be used")
    parser.add_argument("--bert",default="TurkuNLP/bert-base-finnish-cased-v1",help="BERT to use. Default: %(default)s")
    args = parser.parse_args()

    # The text to go with the index...
    indexed_sents=None
    if args.indexed_sentences:
        indexed_sents=[]
        with gzip.open(args.indexed_sentences) as f_in:
            for line in f_in:
                line=line.strip()
                indexed_sents.append(line)


    index=faiss.read_index(args.prefilled_index)
    index.nprobe=12
    res=faiss.StandardGpuResources()
    index_gpu=faiss.index_cpu_to_gpu(res,0,index)
    print("FAISS loaded and on GPU",file=sys.stderr,flush=True)
    index_gpu.nprobe=12 #nearest cells to look for

    bert_tokenizer=transformers.BertTokenizer.from_pretrained(args.bert)
    print("Load model",file=sys.stderr,flush=True)
    bert_model=transformers.BertModel.from_pretrained(args.bert).eval().cuda()

    s_dataset=embed_data.SentenceDataset(sys.stdin,bert_tokenizer)
    s_datareader=embed_data.fluid_batch(s_dataset,12000)
    with torch.no_grad():
        for batch in s_datareader:
            emb_src=embed.embed_batch(batch,bert_model)
            for k in (600,800,1200,1400,2000):
                W,I=index_gpu.search(emb_src.cpu().numpy(),k=5) #k=5
                for src_i,tgt_ws,tgt_is in zip(batch["line_idx"],W,I):
                    print(src_i,tgt_ws[:3],tgt_is[:3])

            #    pickle.dump((batchfile,W,I),f_out,pickle.HIGHEST_PROTOCOL)
            #    print("Shape",batchfile,W.shape,I.shape,file=sys.stderr,flush=True)

