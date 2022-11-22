import faiss
import torch
import random
import tqdm
import pickle
import sys


def random_sample_of_batches(batch_files, proportion):
    """Takes a random sample of batches from all batch_files
    this is used to make training data for faiss. Proportion
    is from [0,1] interval"""
    all_batches = []
    batch_files = list(batch_files)
    random.shuffle(batch_files)
    with tqdm.tqdm() as pbar:
        for b in batch_files:
            with open(b, "rb") as f:
                while True:
                    try:
                        sent_idx, embedding_batch = pickle.load(f)
                        # do I want to keep it?
                        if random.random() < proportion:
                            all_batches.append(embedding_batch)
                        pbar.update(embedding_batch.shape[0])
                    except:  # no more batches
                        break
    random.shuffle(all_batches)
    print("Got", len(all_batches), "random batches", file=sys.stderr, flush=True)
    return torch.vstack(all_batches)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("BATCHFILES", default=None, nargs="+",
                        help="Batch files saved by embed.py")
    parser.add_argument("--prepare-sample", default=None,
                        help="File name to save the sampled examples to. Prepares sample from batchfiles on which faiss can be trained. Does a 5% sample by default.")
    parser.add_argument("--train-faiss", default=None,
                        help="File name to save the trained faiss index to. BATCHFILES should be a single .pt produced by --prepare-sample")
    parser.add_argument("--fill-faiss", default=None, help="Fill faiss index with vectors and save to index with the name given i this argument. BATCHFILES are all batchfiles to store into the index (will be sorted by name). Give the name of the trained index (trained with --train-faiss) in the argument --pretrained-index")
    parser.add_argument("--pretrained-index", default=None,
                        help="Name of the pretrained index to be used for --fill-fais")
    args = parser.parse_args()

    if args.prepare_sample:
        sampled = random_sample_of_batches(sorted(args.BATCHFILES), 0.1)
        torch.save(sampled, args.prepare_sample)
    elif args.train_faiss:
        assert len(
            args.BATCHFILES) == 1, "Give one argument which is a .pt file produced by --prepare-sample"

        quantizer = faiss.IndexFlatL2(768)
        # 768 is bert size, 1024 is how many Voronoi cells we want, 12 is number of quantizers, and these are 8-bit
        index = faiss.IndexIVFPQ(quantizer, 768, 1024, 12, 8)
        res = faiss.StandardGpuResources()
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index)

        sampled_vectors = torch.load(args.BATCHFILES[0])
        print("Training on", sampled_vectors.shape, "vectors", flush=True)

        # how comes this doesnt take any time at all ...?
        index_gpu.train(sampled_vectors.numpy())
        print("Done training", flush=True)
        trained_index = faiss.index_gpu_to_cpu(index_gpu)
        faiss.write_index(trained_index, args.train_faiss)
    elif args.fill_faiss:
        index = faiss.read_index(args.pretrained_index)
        res = faiss.StandardGpuResources()
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index)
        all_batches = list(sorted(args.BATCHFILES))
        for batchfile in tqdm.tqdm(all_batches):
            with open(batchfile, "rb") as f:
                while True:
                    try:
                        line_idx, embedded_batch = pickle.load(f)
                        index_gpu.add_with_ids(
                            embedded_batch.numpy(), line_idx.numpy())
                    except EOFError:
                        break  # no more batches in this file

        index_filled = faiss.index_gpu_to_cpu(index_gpu)
        faiss.write_index(index_filled, args.fill_faiss)
        print("Index has", index_filled.ntotal, "vectors. Done.")
