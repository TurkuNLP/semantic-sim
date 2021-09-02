import pytorch_lightning as pl
import gzip
import itertools
import transformers
import torch
import os
import sys
from torch.utils.data import DataLoader, IterableDataset

def open_possibly_gz_file(f_or_name):
    if f_or_name is None:
        return None
    elif isinstance(f_or_name,str): #a file name of some sort
        if f_or_name.endswith(".gz"):
            return gzip.open(f_or_name,"rt")
        elif os.path.exists(f_or_name):
            return open(f_or_name)
        elif os.path.exists(f_or_name+".gz"):
            return gzip.open(f_or_name+".gz","rt")
        else:
            raise ValueError(f"No such file {f_or_name}, neither plain nor zipped")
    else:
        return f_or_name

class SentenceDataset(IterableDataset):

    def __init__(self,f_in,bert_tokenizer,thisjob=0,jobs=1):
        """f_in is either filename or open file"""
        self.data_src=open_possibly_gz_file(f_in)
        self.bert_tokenizer=bert_tokenizer
        self.thisjob=thisjob
        self.jobs=jobs

    def prep_text_sequence(self,txt):
        tok=self.bert_tokenizer.convert_tokens_to_ids(self.bert_tokenizer.tokenize(txt))[:510] #CUT TO 510
        enc=self.bert_tokenizer.build_inputs_with_special_tokens(tok) #<CLS> ... <SEP> as integers, max length is 512 then
        spec_token_mask=self.bert_tokenizer.get_special_tokens_mask(enc,already_has_special_tokens=True)
        attention_mask=[1]*len(enc)
        token_type_id=[0]*len(enc)
        return tok,enc,spec_token_mask,attention_mask,token_type_id
        
    def yield_tokenized_sentences(self):
        """bert-tokenize and encode sentences from the data, yield as dictionaries"""
        for line_idx,line_src in enumerate(self.data_src):
            if line_idx % self.jobs == self.thisjob: #this line is mine!
                line_src=line_src.rstrip("\n")
                data_item={"line_idx":line_idx}
                tok,enc,spec_token_mask,attention_mask,token_type_id=self.prep_text_sequence(line_src)
                data_item["enc"]=enc
                data_item["spec_token_mask"]=spec_token_mask
                data_item["attention_mask"]=attention_mask
                data_item["token_type_id"]=token_type_id

                yield data_item

    def __iter__(self):
        return self.yield_tokenized_sentences()


def collate(itemlist):
    """Receives a batch in making. It is a list of dataset items, which are themselves dictionaries with the keys as returned by the dataset
    since these need to be zero-padded, then this is what we should do now. Is an argument to DataLoader"""
    batch={}
    for k in "enc","attention_mask","spec_token_mask", "token_type_id":
        batch[k]=pad_with_zero([item[k] for item in itemlist])
    batch["line_idx"]=torch.LongTensor([item["line_idx"] for item in itemlist])
    return batch

def pad_with_zero(vals):
    vals=[torch.LongTensor(v) for v in vals]
    padded_vals=torch.nn.utils.rnn.pad_sequence(vals,batch_first=True)
    return padded_vals
    
def fluid_batch(from_data,max_element_count):
    """A little replacement of DataLoader, builds batches with variable length and caps the maximum element count in each batch
    taking into account padding. So it can build a long batch of short sequences, or a short batch of long sequences.
    This is extremely useful to keep GPU memory utilization high!"""
    current_batch=[]
    current_max_len=0
    for item in from_data:
        src_len=len(item["enc"])
        #what would the new size of the batch be?
        max_len=max((current_max_len,src_len))
        elems=max_len*(len(current_batch)+1)
        if elems>max_element_count: #nope gotta yield first or else I blow the max
            yield collate(current_batch)
            current_batch=[]
            current_max_len=0
        current_batch.append(item)
        current_max_len=max((current_max_len,src_len))
    else:
        if current_batch:
            yield collate(current_batch)

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--bert-model", default=None, help="BERT model name or path")
    args = parser.parse_args()
    bert_tokenizer=transformers.BertTokenizer.from_pretrained(args.bert_model)
    s_dataset=SentenceDataset(sys.stdin,bert_tokenizer)
    s_datareader=DataLoader(s_dataset,collate_fn=collate,batch_size=60)
    for x in s_datareader:
        print(x)
        break
    
