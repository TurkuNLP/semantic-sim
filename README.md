# Faiss + BERT/SBERT based indexing and search

## 1. Gather unique sentences

The overall input to this pipeline is a list of **unique** sentences to embed and index. This is produced by these:

* `gather_uniq_sents.sh` produces the file `all_data_pos_uniq.gz` with unique, clean(-ish) sentences
* `clean_data.py` does elementary cleanup by droping too short, too long, or too weird sentences

## 2. Index sentences for fast lookup text->index

We need to be able to look up a sentence position (index) in `all_data_pos_uniq.gz` by its text. This is done by calculating a 15-byte hash of the sentence (there seem to be no collisions at this length on our 400M sentence data)

* `index_texts.sbatch` is the script to run indexing
* `index_sentences.py` implements the indexing, and produces a single large pickle file which is a dictionary of sentence-text -> position index as integer


## 3. Index sentences for fast lookup index->text

We will need to be able to print the sentences based on their position (index) in `all_data_pos_uniq.gz`. This step implements a simple indexing for fast random-access to the sentence file. No gzipping, so this consumes about as much space as the unzipped `all_data_pos_uniq.gz`

* `zcat all_data_pos_uniq.gz | python3  mmap_index.py --index-lines-to all_data_pos_uniq` this produces several files `all_data_pos_uniq.data`, `all_data_pos_uniq.index`, `all_data_pos_uniq.lengths`, `all_data_pos_uniq.meta` which can be used to get sentence text based on its ID

## 4. Calculate the embeddings

The embeddings of each sentence in `all_data_pos_uniq.gz` needs to be calculated in a distributed fashion. These are stored into iterable pickle files, one file per parallel embedding process. The input is a single file, and when running an individual `embed_data.py` job, it is told which is its rank and what is the total number of jobs. So if there are 30 processes, rank 1 process takes lines 0, 30, 60; rank 2 proceess takes 1,31,61,... etc. For the 400M sentences, this produces about 1.2TB of embeddings.

* `embed.py` produces BERT embeddings, `embed_sbert.py` produces SBERT embeddings
* `embed_data.py` is joint for the above two and does dynamic batching etc
* `embed_sbatch.sh` is a script (slurm also) which runs a single process of `embed_sbert.py`
* `gen_embed_sbatch.sh` generates a number of `sbatch` commands which submit the slurm jobs for the parallel runs

## 5. 


