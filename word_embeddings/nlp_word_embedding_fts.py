import numpy as np
from scipy import spatial
import tqdm
import os
import re


def load_embeddings(filename, dimensions):
    print("Loading pre-trained embeddings " + filename + "...")
    embeddings_dict = {}
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            word = ""
            i = 0
            while i < len(values) - dimensions:
                word += values[i] + " "
                i += 1
            word = word.strip()
            vector = np.asarray(values[i:], dtype=np.float32)
            if len(vector) == 300:
                embeddings_dict[word] = vector
    print("Loaded.")
    return embeddings_dict


def load_docs(filename):
    print("Loading document embeddings " + filename + "...")
    embeddings_dict = {}
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split(';')
            vector = np.asarray(values[1].split(), dtype=np.float32)
            embeddings_dict[values[0]] = vector
    print("Document embeddings dictionary loaded.")
    return embeddings_dict


def load_top_100(filename):
    dicti = {}
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            vals = line.split()
            if vals[0] in dicti:
                dicti[vals[0]].append(vals[2])
            else:
                dicti[vals[0]] = [vals[2]]
    print("Top 100 dictionary loaded")
    return dicti


def load_queries(line, dct):
    query = line.split('\t')
    query_tokens = query[1].replace('/', ' ').replace('\\', ' ').replace('-', ' ').replace(',', ' ').replace(':', ' ').replace('.', ' ').split()
    query_term_embeds = []
    for token in query_tokens:
        embed = dct.get(token)
        if embed is not None and np.isfinite(embed).all():
            query_term_embeds.append(embed)
        else:
            alt = dct.get(re.sub("[^0-9a-zA-Z]+", "", token))
            if alt is not None and np.isfinite(alt).all():
                query_term_embeds.append(alt)
    return query[0], query_term_embeds


def compute_distances(dct, doc_dct, topdocs, queries, output):
    print("Writing computed distances to "+output+"...")
    default = 0.5
    with tqdm.tqdm(total=os.path.getsize(queries)) as pbar:
        with open(queries, 'r', encoding='utf-8') as f:
            with open(output, 'w', encoding='utf-8') as o:
                for line in f:
                    pbar.update(len(line.encode('utf-8')))
                    qid, query_term_embeds = load_queries(line, dct)
                    if not query_term_embeds or len(query_term_embeds) == 0:
                        print("No valid feature vals for query " + line + "), defaulting to " +
                              "cosine distance " + str(default))
                        doc_ids = topdocs.get(qid)
                        if doc_ids is None or not doc_ids or None in doc_ids:
                            print("Also no top documents for query "+line+", skipping this query!")
                        else:
                            for d in doc_ids:
                                o.write(qid + "\t" + d + "\t" + str(default) + '\n')
                    else:
                        doc_ids = topdocs.get(qid)
                        if doc_ids is None or not doc_ids or None in doc_ids:
                            print("No top documents for query "+line)
                        else:
                            doc_embs = []
                            for d in doc_ids:
                                doc_embs.append(doc_dct.get(d))

                            try:
                                distances = spatial.distance.cdist(np.array(doc_embs),
                                                                   np.array(query_term_embeds), 'cosine').mean(1)
                                for i in range(0, len(doc_embs)):
                                    o.write(qid + "\t" + doc_ids[i] + "\t" + str(distances[i]) + '\n')
                            except ValueError as ve:
                                print(ve)
                                if "XA" in ve:
                                    print(format(doc_embs))
                                elif "XB" in ve:
                                    print("Something went wrong with query "+line)
                                    print(format(query_term_embeds))


if __name__ == "__main__":
    files = [
    {"queries": "/home/lucile/Documents/nlp/train/msmarco-doctrain-queries.tsv",
    "documents": "/home/lucile/Documents/nlp/train/msmarco-doctrain-top100"},
    {"queries": "/home/lucile/Documents/nlp/val/msmarco-docdev-queries.tsv",
    "documents": "/home/lucile/Documents/nlp/val/msmarco-docdev-top100"},
    {"queries": "/home/lucile/Documents/nlp/test/msmarco-test2019-queries.tsv",
    "documents": "/home/lucile/Documents/nlp/test/msmarco-doctest2019-top100"}]
    doc_embeddings = [{'name': "glove", 'loc': "/home/lucile/Documents/nlp/corpus/msmarco-docs-embeddings-glove.txt",
                       'emb_loc': '/home/lucile/Documents/nlp/wordvectors/glove.840B.300d.txt',
                       'dims': 300},
                      {'name': "fasttext",
                       'loc': "/home/lucile/Documents/nlp/corpus/msmarco-docs-embeddings-fasttext.txt",
                       'emb_loc': '/home/lucile/Documents/nlp/wordvectors/crawl-300d-2M.vec',
                       'dims': 300}]

    for tup in doc_embeddings:
        dictionary = load_embeddings(['emb_loc'], tup['dims'])
        docs_dictionary = load_docs(tup['loc'])
        for pair in files:
            compute_distances(dictionary, docs_dictionary, pair['documents'], pair['queries'],
                              pair['queries'][0:len(pair['queries']) - 7] + "y-doc-fts-" + tup['name'] + ".tsv")
