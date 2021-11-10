FROM python:3.9
WORKDIR /usr/src/app

RUN pip3 install --upgrade pip

COPY requirements.txt ./

RUN pip3 install --upgrade --no-cache-dir -r requirements.txt

COPY *.py ./
COPY templates/ ./templates
COPY static/ ./static

ENV FLASK_APP demo
ENV APP_ROOT /sbert400m
ENV SBERT_TOKENIZER TurkuNLP/bert-base-finnish-cased-v1
ENV SBERT_MODEL /datamount/sbert-cased-finnish-paraphrase
ENV MMAP_SFILENAME /datamount/all_data_pos_uniq
ENV FAISS_IDX_FILENAME /datamount/faiss_index_filled_sbert.faiss
EXPOSE 8866

CMD python3 -m flask run --host 0.0.0.0 --port 8866
