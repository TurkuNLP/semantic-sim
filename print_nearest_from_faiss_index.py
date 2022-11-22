import torch
import transformers
import gzip
import tqdm
import sys
import os
import pickle
import itertools
import embed_data
import embed
import faiss
from sentence_transformers import SentenceTransformer


def open_possibly_gz_file(f_or_name):
    if f_or_name is None:
        return None
    elif isinstance(f_or_name, str):  # a file name of some sort
        if f_or_name.endswith(".gz"):
            return gzip.open(f_or_name, "rt")
        elif os.path.exists(f_or_name):
            return open(f_or_name)
        elif os.path.exists(f_or_name+".gz"):
            return gzip.open(f_or_name+".gz", "rt")
        else:
            raise ValueError(
                f"No such file {f_or_name}, neither plain nor zipped")
    else:
        return f_or_name


def grouper(iterable, n, fillvalue=None):
    "Collect data into non-overlapping fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    #parser.add_argument("--sentencefiles",nargs="+",help="Files with the sentences in the same order as in the index. Can be (and probably is) gz files")
    parser.add_argument("--faiss-index", help="FAISS index file")
    parser.add_argument("--bert-model", help="Name of the bert model to use")
    parser.add_argument("--sbert-model", help="Path of the sbert model to use")
    parser.add_argument("--sbert-tokenizer",
                        help="Path of the sbert tokenizer to use")
    parser.add_argument("--out", help="Pickle with the nearest neighbors")
    args = parser.parse_args()

    # #Read all texts first, we will need these to be able to print the outcome
    # all_sents=[]
    # for fname in tqdm.tqdm(args.sentencefiles):
    #     f=open_possibly_gz_file(fname)
    #     all_sents.extend((l.strip().encode("utf-8") for l in f if l.strip()))
    #     f.close()
    # print(f"Read {len(all_sents)} sentences",file=sys.stderr,flush=True)
    if args.bert_model:
        print("Load tokenizer model", file=sys.stderr, flush=True)
        bert_tokenizer = transformers.BertTokenizer.from_pretrained(
            args.bert_model)
        print("Load model", file=sys.stderr, flush=True)
        bert_model = transformers.BertModel.from_pretrained(
            args.bert_model).eval().cuda()
        print("Done loading", file=sys.stderr, flush=True)
    elif args.sbert_model:
        print("Load tokenizer model", file=sys.stderr, flush=True)
        bert_tokenizer = transformers.AutoTokenizer.from_pretrained(
            args.sbert_tokenizer)
        print("Load model", file=sys.stderr, flush=True)
        sbert_model = SentenceTransformer(args.sbert_model).eval().cuda()
        print("Done loading", file=sys.stderr, flush=True)

    s_dataset = embed_data.SentenceDataset(sys.stdin, bert_tokenizer, 0, 1)
    # DataLoader(sp_dataset,collate_fn=embed_data.collate,batch_size=15)
    s_datareader = embed_data.fluid_batch(s_dataset, 6000)

    index = faiss.read_index(args.faiss_index)
    index.nprobe = 12  # ah jesus what was this doing again?
    res = faiss.StandardGpuResources()
    index_gpu = faiss.index_cpu_to_gpu(res, 0, index)

    with tqdm.tqdm() as pbar, open(args.out, "wb") as fout, torch.no_grad():
        for batch in s_datareader:
            if args.bert_model:
                emb = embed.embed_batch(batch, bert_model).cpu().numpy()
            else:
                emb = sbert_model.encode(batch["txt"])
            W, I = index_gpu.search(emb, 10)
            pickle.dump((batch["line_idx"], I), fout)
            pbar.update(emb.shape[0])

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
