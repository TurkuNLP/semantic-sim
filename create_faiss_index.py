import faiss
import torch
import glob
import random
import tqdm

def random_sample_of_batches(batch_files,proportion):
    """Takes a random sample of batches from all batch_files
    this is used to make training data for faiss. Proportion is from [0,1] interval"""
    all_batches=[]
    for b in tqdm(batch_files):
        batches=torch.load(b)
        random.shuffle(batches)
        batches=batches[:int(len(batches)*proportion)]
        all_batches.extend(batches)
    return torch.vstack(all_batches)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("BATCHFILES",default=None,nargs="+",help="Batch files saved by embed.py")
    parser.add_argument("--prepare-sample",default=None,help="File name to save the sampled examples to. Prepares sample from batchfiles on which faiss can be trained. Does a 5% sample by default.")
    args = parser.parse_args()


    if args.prepare_sample:
        sampled=random_sample_of_batches(sorted(args.BATCHFILES),0.05)
        torch.save(sampled,args.prepare_sample)
        

# devs=read_dev_vectors()

# #Training

# quantizer=faiss.IndexFlatL2(768)
# index=faiss.IndexIVFPQ(quantizer,768,1024,48,8)
# res=faiss.StandardGpuResources()
# index_gpu=faiss.index_cpu_to_gpu(res,0,index)
# index_gpu.train(devs.numpy())
# index=faiss.index_gpu_to_cpu(index_gpu)
# faiss.write_index(index,"trained.faiss")


# #TEST

# index=faiss.read_index("trained.faiss")
# res=faiss.StandardGpuResources()
# index_gpu=faiss.index_cpu_to_gpu(res,0,index)
# index_gpu.add(devs.numpy())
# print(index_gpu.search(devs.numpy()[:20],5))
# index=faiss.index_gpu_to_cpu(index_gpu)
# faiss.write_index(index,"dev.faiss")

