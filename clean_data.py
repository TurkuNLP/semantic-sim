import re
import sys
import multiprocessing
import time
import gzip
import tqdm

def remove_initial_dash(s): #removes "- "
    return re.sub("^\s*-+\s*","",s)

def line_about_ok(s):
    n_rough_words=len(s.split())
    if n_rough_words<3 or n_rough_words>50: #Let's dump anything shorter than 3 words and longer than 50
        return False, "too few words"
    n_punct_and_junk=len(re.sub("""[a-zA-ZäöåÅÄÖ0-9:"`'(”"',.!?() -]+""","",s))
    if n_punct_and_junk/len(s)>0.1: #Dump stuff with more than 10%
        return False, "too many punct"
    n_num=len(re.sub("[^0-9]+"," ",s).split())
    if n_num>5 and n_num/len(s)>0.2: #Dump stuff with more numbers than 20%
        return False, "too many nums"
    #if not re.match("""^[a-zA-ZäöåÄÖÅ0-9"`'(”]""",s):
    #    return False, "starts with nonalpha" #these looked like junk mostly
    return True, None

def line_clean(line):
    line=line.strip()
    if not line:
        return None
    line=remove_initial_dash(line)
    ok,reason=line_about_ok(line)
    if ok:
        return True, "ok", line.strip()
    else:
        return False, reason, line.strip()
    

def yield_line_bunch(inp):
    buff=[]
    for line in tqdm.tqdm(inp):
        buff.append(line)
        if len(buff)>1000000:
            yield buff
            buff=[]
    else:
        if buff:
            yield buff

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--outfile-pos",default=None,help="Where the good stuff goes")
    parser.add_argument("--outfile-neg",default=None,help="Where the neg stuff goes")
    args = parser.parse_args()


    
    start=time.time()
    counter=0
    dumped=0
    with gzip.open(args.outfile_pos,"wt") as fout_pos, gzip.open(args.outfile_neg,"wt") as fout_neg:
        for lines in yield_line_bunch(sys.stdin):
            print("FEEDING BATCH OF",len(lines),"LINES",file=sys.stderr,flush=True)
            with multiprocessing.Pool(processes=18) as p:
                for s in p.imap_unordered(line_clean,lines,chunksize=10000):
                    if s is None:
                        continue
                    ok,reason,s_text=s
                    if ok:
                        counter+=1
                        print(s_text,file=fout_pos)
                        if counter%100000==0:
                            print("Line",counter,"  (",counter/(time.time()-start),")   ","Dumped",dumped,file=sys.stderr,flush=True)
                            fout_pos.flush()
                            fout_neg.flush()
                    elif not ok:
                        dumped+=1
                        print(reason,s_text,file=fout_neg,sep="\t")
