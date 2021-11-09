import flask
from flask import Flask
from flask import render_template, request
import re
import os
import interactive_index
    
app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True
APP_ROOT = os.environ.get("APP_ROOT","/")
app.config["APPLICATION_ROOT"] = APP_ROOT

sbert_tokenizer_name = os.environ.get("SBERT_TOKENIZER","TurkuNLP/bert-base-finnish-cased-v1")
sbert_model_name = os.environ.get("SBERT_MODEL","/scratch/project_2000539/pb_faiss/sbert-cased-finnish-paraphrase")
mmap_sentence_filename = os.environ.get("MMAP_SFILENAME","/scratch/project_2000539/pb_faiss/all_data_pos_uniq")
faiss_index_fname = os.environ.get("FAISS_IDX_FILENAME","/scratch/project_2000539/pb_faiss/faiss_index_filled_sbert.faiss")

nn_qry=interactive_index.IDemoSBert(sbert_tokenizer_name,sbert_model_name,faiss_index_fname,mmap_sentence_filename)

@app.route("/")
def index():
    return render_template("index.html",app_root=APP_ROOT)

@app.route("/predict",methods=["POST"])
def predict():
    global nn_qry
    inpsentence=request.json["sentencein"].strip()
    res=nn_qry.knn([inpsentence])
    nearest=[]
    for sent,hits in res:
        for score,h in hits:
            nearest.append(h)

    return {"predictions_html":render_template("result.html",knns=nearest)}
