import sys
import argparse
import pickle
from index_sentences import s2hash
import re

def remove_initial_dash(s): #removes "- "
    return re.sub("^\s*-+\s*","",s)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--s2i', help='s2i hash to idx dict pickle, result of index_sentences.py')
    args = parser.parse_args()

    with open(args.s2i,"rb") as f:
        sdict=pickle.load(f)
    
    print(f"Loaded dict of {len(sdict)} hashes")

    found=0
    lost=0
    for idx,qline in enumerate(sys.stdin):
        qline=qline.strip()
        qline=remove_initial_dash(qline)
        _,linehash=s2hash((0,qline)) #this function takes a pair (id,sent)
        if linehash in sdict:
            found+=1
        else:
            lost+=1
            print("LOST",qline)
    
    print(f"Found {found} lost {lost}")
