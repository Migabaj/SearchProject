import os
import json
import time
import torch
import pickle
import gensim
import pymorphy2
import numpy as np
from scipy.sparse import load_npz
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import Normalizer
from flask import Flask, render_template, request

curr_dir = os.getcwd()
morph = pymorphy2.MorphAnalyzer()
norm = Normalizer()
app = Flask(__name__)

print(curr_dir)
fasttext_model = gensim.models.KeyedVectors.load("models/fasttext/araneum_none_fasttextcbow_300_5_2018.model")
print("tokenizer...")
bert_tokenizer = AutoTokenizer.from_pretrained("models/bert/sbert_large_nlu_ru")
print("model...")
bert_model = AutoModel.from_pretrained("models/bert/sbert_large_nlu_ru")

with open(os.path.join(curr_dir, "data/corpus.json"), encoding="utf-8") as f:
    corpus = json.load(f)

with open(os.path.join(curr_dir,
                       "data/count/embeddings.npz"),
          "rb") as frb:
    count_embeddings = load_npz(frb)
len_d = count_embeddings.sum(axis=1)
avdl = len_d.mean()

with open(os.path.join(curr_dir,
                       "data/count/embeddings_normalized.npz"),
          "rb") as frb:
    count_norm_embeddings = load_npz(frb)

with open(os.path.join(curr_dir,
                       "data/tfidf/embeddings.npz"),
          "rb") as frb:
    tfidf_embeddings = load_npz(frb)

with open(os.path.join(curr_dir,
                       "data/tfidf/embeddings_noidf.npz"),
          "rb") as frb:
    tf_embeddings = load_npz(frb)

with open(os.path.join(curr_dir,
                       "data/bm25/embeddings.npz"),
          "rb") as frb:
    bm25_embeddings = load_npz(frb)

bert_norm_embeddings = torch.load(os.path.join(curr_dir,
                                               "data/bert/embeddings_normalized.pt"))

fasttext_norm_embeddings = np.load(os.path.join(curr_dir,
                                                "data/fasttext/embeddings_normalized.npy"))


with open(os.path.join(curr_dir,
                       "data/count/vectorizer.pckl"),
          "rb") as frb:
    count_vectorizer = pickle.load(frb)

with open(os.path.join(curr_dir,
                       "data/tfidf/vectorizer.pckl"),
          "rb") as frb:
    tfidf_vectorizer = pickle.load(frb)

with open(os.path.join(curr_dir,
                       "data/tfidf/vectorizer_noidf.pckl"),
          "rb") as frb:
    tf_vectorizer = pickle.load(frb)


def preprocess(text):
    tokens = word_tokenize(text)
    tokens = [morph.parse(tok)[0].normal_form for tok in tokens
              if tok not in stopwords.words("russian")]
    " ".join(tokens).lower()
    return " ".join(tokens).lower()


def search(query, corpus, method):
    if method == "CountVectorizer":
        global count_embeddings
        query_embedded = norm.transform(count_vectorizer.transform([query]))
        cos_sims = np.dot(query_embedded, count_norm_embeddings.T).toarray()
        args = np.argsort(cos_sims, axis=1).ravel()[:-6:-1]
        return np.array(corpus)[args]

    elif method == "TfidfVectorizer":
        global tfidf_vectorizer
        query_embedded = tfidf_vectorizer.transform([query])
        cos_sims = np.dot(query_embedded, tfidf_embeddings.T).toarray()
        args = np.argsort(cos_sims, axis=1).ravel()[:-6:-1]
        return np.array(corpus)[args]

    elif method == "BM25":
        query_embedded = count_vectorizer.transform([query])
        sims = np.dot(query_embedded, bm25_embeddings.T).toarray()
        args = np.argsort(sims, axis=1).ravel()[:-6:-1]
        return np.array(corpus)[args]

    elif method == "FastText":
        query_tokenized = gensim.utils.tokenize(query)
        query_embedded = norm.transform(np.expand_dims(np.mean(fasttext_model[query_tokenized],
                                                axis=0),
                                        axis=0))
        cos_sims = np.squeeze(np.dot(query_embedded, fasttext_norm_embeddings.T))
        args = np.argsort(cos_sims, axis=0).ravel()[:-6:-1]
        return np.array(corpus)[args]

    elif method == "BERT":
        query_tokenized = bert_tokenizer([query],
                                         padding=True,
                                         truncation=True,
                                         max_length=24,
                                         return_tensors='pt')
        with torch.no_grad():
            model_query_output = bert_model(**query_tokenized)
        query_embedded = norm.transform(model_query_output[0][:, 0])
        cos_sims = np.squeeze(np.dot(query_embedded, bert_norm_embeddings.T))
        args = np.argsort(cos_sims, axis=0).ravel()[:-6:-1]
        return np.array(corpus)[args]


@app.route('/')
def search_engine():
    return render_template("main.html")


@app.route('/search', methods=['POST'])
def results():
    method = request.form["methods"]
    print(method)
    start = time.time()
    query = preprocess(request.form['query'])
    answers = search(query, corpus, method)
    total_time = time.time() - start
    return render_template("search.html", answers=answers, total_time=total_time)


if __name__ == '__main__':
    app.run()
