# Faiss + BERT/SBERT based indexing and search

This is code for the NN experiment in the paraphrase paper. It is capable of embedding and indexing 400M sentences, and doing fast lookup among these to establish how often paraphrases are found in each others' nearest neighbor lists. This readme documents the order in which these steps should run and how the experiment is built.

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

## 5. Build the FAISS index

This step takes the embeddings from step 4 and pushes them into FAISS index.

* `create_faiss_index.py` builds the index, it needs to happen in three steps: a) sample data on which the FAISS index' quantizer is trained b) create an index pre-trained with this data c) fill the index with all data
* `create_faiss_sbatch.sh` carries out these steps and ends up producing `faiss_index_filled_{bert|sbert}.faiss` files

## 6. Use the FAISS index to query for nearest neighors

* `print_nearest_from_faiss_index.py` takes on input any number of sentences, embeds them with BERT or SBERT and uses the faiss index from step 5 to get their nearest neighbors. These are then output as pickled tensors with the ids of the nearest neighbors, by default the code asks for 2048 NNs which is the max allowed on GPU
* `query_sents_sbatch.sh` the slurm script for this

## 7. Print the nearest neighbors

* `id2txt.py` takes a) the input file for (6); b) the id2text sentence index from step (3); c) the nearest neighbor output from step (6); and prints all of these as texts, not ids; this makes it possible to check the sanity of the output and post-process it any way you like

## 8. The paraphrase lookup experiment

Establishing how often a paraphrase pair is found and on what rank. Given parapharse data (sent1,sent2,label), this code outputs how often `sent2` is found when querying with `sent1`. The assumptions here is that these were included in the original data. But because the original data is produced with sort and uniq, and there is plenty of it, we don't necessarily know at which index `sent1` and `sent2` reside.

* `run_nn_experiment.py` is first run with these parameters: a) query file which is the texts of the query sentences used in step (6) ie when building the nearest neighbor lists; b) the text2index index from step (2); c) paraphrase data with the pairs in the Turku Paraphrase corpus format. When run like this, it outputs the paraphrase data with additional keys in the json that contain the indices into `all_data_pos_uniq.gz` and the query file
* `run_nn_experiment.py` is then run with these parameters: a) the output of the previous step, basically a paraphrase json file with information about where to find what; b) the nearest neighbor output from step (6); c) the fast mmap id->text lookup from step (3); and it output the statistics and prints the actual sentences for the NN experiment


