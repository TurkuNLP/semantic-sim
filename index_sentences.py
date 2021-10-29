from sqlitedict import SqliteDict
import tqdm
import argparse
import sys
import multiprocessing
import hashlib
import pickle

def yield_line_blocks(inp):
    cache=[]
    for idx,line in enumerate(inp):
        line=line.strip()
        cache.append((idx,line))
        if len(cache)>10000000:
            yield cache
            cache=[]
    else:
        if cache:
            yield cache

def s2hash(idx_sent):
    idx,sent=idx_sent
    return idx,hashlib.sha1(sent.encode("utf-8")).hexdigest()[:15]

if __name__=="__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--out-s2i', help='s2i sqlitedict')
    args = parser.parse_args()

    s2i={}
    
    for lines_block in yield_line_blocks(sys.stdin):
        with multiprocessing.Pool(8) as pool:
            for idx,linehash in tqdm.tqdm(pool.imap_unordered(s2hash,lines_block,chunksize=100000)):
                if linehash in s2i:
                    print("CONFLICT",flush=True)
                s2i[linehash]=idx
    with open(args.out_s2i,"wb") as f:
        pickle.dump(s2i,f)
