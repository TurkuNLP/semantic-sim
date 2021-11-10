import torch
import transformers
import glob
import random
import gzip
import tqdm
import sys
import pickle
import numpy
import traceback
import itertools
import embed_data
import embed
import faiss
from sentence_transformers import SentenceTransformer
import mmap_index

class IDemo:

    def __init__(self,faiss_index_fname,mmap_sentence_filename,gpu=False):
        self.index=faiss.read_index(faiss_index_fname)
        self.index.nprobe=12 #ah jesus what was this doing again?
        if gpu:
            res=faiss.StandardGpuResources()
            self.index=faiss.index_cpu_to_gpu(res,0,self.index)
        self.mmidx=mmap_index.Qry(mmap_sentence_filename)

    def embed(self,sentlist):
        raise NotImplementedError("You gotta define embed() in a subclass")

    def knn(self,sentlist):
        res=[]
        W,I=self.embed(sentlist)
        for sent,ws,nns in zip(sentlist,W,I):
            sent_res=[]
            for w,nn in zip(ws,nns):
                nn_sent=self.mmidx.get(nn)
                sent_res.append((w,nn_sent))
            res.append((sent,sent_res))
        return res



class IDemoSBert(IDemo):

    def __init__(self,sbert_tokenizer_name,sbert_model_name,faiss_index_fname,mmap_sentence_filename,gpu=False):
        super().__init__(faiss_index_fname,mmap_sentence_filename,gpu)
        bert_tokenizer=transformers.BertTokenizer.from_pretrained(sbert_tokenizer_name)
        self.model=SentenceTransformer(sbert_model_name).eval()
        if gpu:
            self.model=self.model.cuda()

    def embed(self,sentlist):
        print("Starting encoding",file=sys.stderr,flush=True)
        emb=self.model.encode(sentlist)
        print("Starting index search",file=sys.stderr,flush=True)
        W,I=self.index.search(emb,32)
        print("Done index search",file=sys.stderr,flush=True)
        return W,I



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

def grouper(iterable, n, fillvalue=None):
    "Collect data into non-overlapping fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument("--sentencefiles",nargs="+",help="Files with the sentences in the same order as in the index. Can be (and probably is) gz files")
    parser.add_argument("--faiss-index",help="FAISS index file")
    parser.add_argument("--bert-model",help="Name of the bert model to use")
    parser.add_argument("--sbert-model",help="Path of the sbert model to use")
    parser.add_argument("--sbert-tokenizer",help="Path of the sbert tokenizer to use")
    parser.add_argument("--mmap-idx",help="Mmap index with the sentence")
    parser.add_argument("--gpu",default=False,action="store_true",help="GPU?")
    args = parser.parse_args()

    idemo=IDemoSBert(args.sbert_tokenizer,args.sbert_model,args.faiss_index,args.mmap_idx,args.gpu)
    res=idemo.knn(["Turussa on kivaa asua, tykkään tästä paikasta."])
    for sent,hits in res:
        print(f"Qry: {sent}")
        for score,h in hits:
            print(f"   Hit: {h}")

    # # #Read all texts first, we will need these to be able to print the outcome
    # # all_sents=[]
    # # for fname in tqdm.tqdm(args.sentencefiles):
    # #     f=open_possibly_gz_file(fname)
    # #     all_sents.extend((l.strip().encode("utf-8") for l in f if l.strip()))
    # #     f.close()
    # # print(f"Read {len(all_sents)} sentences",file=sys.stderr,flush=True)
    # if args.bert_model:
    #     print("Load tokenizer model",file=sys.stderr,flush=True)
    #     bert_tokenizer=transformers.BertTokenizer.from_pretrained(args.bert_model)
    #     print("Load model",file=sys.stderr,flush=True)
    #     bert_model=transformers.BertModel.from_pretrained(args.bert_model).eval().cuda()
    #     print("Done loading",file=sys.stderr,flush=True)
    # elif args.sbert_model:
    #     print("Load tokenizer model",file=sys.stderr,flush=True)
    #     bert_tokenizer=transformers.BertTokenizer.from_pretrained(args.sbert_tokenizer)
    #     print("Load model",file=sys.stderr,flush=True)
    #     sbert_model=SentenceTransformer(args.sbert_model).eval().cuda()
    #     print("Done loading",file=sys.stderr,flush=True)


    # s_dataset=embed_data.SentenceDataset(sys.stdin,bert_tokenizer,0,1)
    # s_datareader=embed_data.fluid_batch(s_dataset,6000)#DataLoader(sp_dataset,collate_fn=embed_data.collate,batch_size=15)

    # index=faiss.read_index(args.faiss_index)
    # index.nprobe=12 #ah jesus what was this doing again?
    # res=faiss.StandardGpuResources()
    # index_gpu=faiss.index_cpu_to_gpu(res,0,index)


    # with tqdm.tqdm() as pbar, open(args.out,"wb") as fout, torch.no_grad():
    #     for batch in s_datareader:
    #         if args.bert_model:
    #             emb=embed.embed_batch(batch,bert_model).cpu().numpy()
    #         else:
    #             emb=sbert_model.encode(batch["txt"])
    #         W,I=index_gpu.search(emb,2048)
    #         pickle.dump((batch["line_idx"],I),fout)
    #         pbar.update(emb.shape[0])


        

    # pbar=tqdm.tqdm(total=len(all_sents))
    # with gzip.open(args.outfile,"wt") as fout:
    #     glob_index=0 #index into row, across all batches, so I can index the sentences
    #     for W,I in yield_batches(args.nnfiles):
    #         for idx,weights,neighbors in zip(range(glob_index,glob_index+len(W)),W,I):
    #             print(f"> {all_sents[idx].decode('utf-8')}",file=fout)
    #             for nn_w,nn_idx in zip(weights[1:],neighbors[1:]): #0 is the sentence itself
    #                 if nn_idx>=len(all_sents):
    #                     continue
    #                 print(f"      {nn_w} {all_sents[nn_idx].decode('utf-8')}",file=fout)
    #             pbar.update(1)
    #         glob_index+=len(W)
