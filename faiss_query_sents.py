import json
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
    parser.add_argument("--para-json",help="If given, query data will come from here and the system will query txt1-txt2 and txt2-txt1 pairs")
    parser.add_argument("-k",default=10,type=int,help="Faiss k. Default: %(default)d")
    args = parser.parse_args()

    # The text to go with the index...
    indexed_sents=None
    if args.indexed_sentences:
        indexed_sents=[]
        with gzip.open(args.indexed_sentences,"rt") as f_in:
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

    if args.para_json:
        def yield_pairs(fjson):
            data=json.load(open(fjson))
            for item in data:
                yield (item["txt1"],item["txt2"])
                yield (item["txt2"],item["txt1"])
                
        s_dataset=embed_data.SentencePairDataset(yield_pairs(args.para_json),bert_tokenizer)
        s_datareader=embed_data.fluid_batch(s_dataset,12000)

        found=0
        total=0
        with torch.no_grad():
            for batch in tqdm.tqdm(s_datareader):
                emb_src=embed.embed_batch(batch,bert_model)
                W,I=index_gpu.search(emb_src.cpu().numpy(),k=args.k)

                #I should now answer; is text2 among the topk hits for text? (text is the first half of the paraphrase, text2 is the second one)
                for src_i,(tgt_ws,tgt_is) in enumerate(zip(W,I)):
                    total+=1
                    hits=[indexed_sents[tgt_i] for tgt_i in tgt_is] #this is the text of the hits
                    if batch["text2"][src_i] in hits: #YAY!
                        found+=1
                        print(batch["text"][src_i])
                        print(batch["text2"][src_i])
                        print()
        print(f"FOUND {found}/{total}={found/total*100}%")
    else:
        s_dataset=embed_data.SentenceDataset(sys.stdin,bert_tokenizer)
        s_datareader=embed_data.fluid_batch(s_dataset,12000)

        with torch.no_grad():
            for batch in s_datareader:
                emb_src=embed.embed_batch(batch,bert_model)
                W,I=index_gpu.search(emb_src.cpu().numpy(),k=args.k)
                for src_i,(tgt_ws,tgt_is) in enumerate(zip(W,I)):
                    print("***",batch["text"][src_i])
                    for tgt_w,tgt_i in zip(tgt_ws,tgt_is):
                        if indexed_sents:
                            print("      ",indexed_sents[tgt_i])

            #    pickle.dump((batchfile,W,I),f_out,pickle.HIGHEST_PROTOCOL)
            #    print("Shape",batchfile,W.shape,I.shape,file=sys.stderr,flush=True)

