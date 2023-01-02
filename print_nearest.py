import os
import gzip
import tqdm
import sys
import pickle


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


def yield_batches(fnames):
    for fname in fnames:
        with open(fname, "rb") as f:
            while True:
                try:
                    _, W, I = pickle.load(f)
                    yield W, I
                except EOFError:
                    break


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--sentencefiles", nargs="+",
                        help="Files with the sentences in the same order as in the index. Can be (and probably is) gz files")
    parser.add_argument("--nnfiles", nargs="+", default=None,
                        help="The nearest neighbor matrices in pickle files produced by faiss_query_all_by_all")
    parser.add_argument("--outfile", default=None, help="outfile.gz")
    args = parser.parse_args()

    # Read all texts first
    all_sents = []
    for fname in tqdm.tqdm(args.sentencefiles):
        f = open_possibly_gz_file(fname)
        all_sents.extend((l.strip().encode("utf-8") for l in f if l.strip()))
        f.close()
    print(f"Read {len(all_sents)} sentences", file=sys.stderr, flush=True)

    pbar = tqdm.tqdm(total=len(all_sents))
    with gzip.open(args.outfile, "wt") as fout:
        glob_index = 0  # index into row, across all batches, so I can index the sentences
        for W, I in yield_batches(args.nnfiles):
            for idx, weights, neighbors in zip(range(glob_index, glob_index+len(W)), W, I):
                print(f"> {all_sents[idx].decode('utf-8')}", file=fout)
                # 0 is the sentence itself
                for nn_w, nn_idx in zip(weights[1:], neighbors[1:]):
                    if nn_idx >= len(all_sents):
                        continue
                    print(
                        f"      {nn_w} {all_sents[nn_idx].decode('utf-8')}", file=fout)
                pbar.update(1)
            glob_index += len(W)
