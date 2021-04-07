import numpy as np
from scipy import spatial

import nlp_load_embeddings as load_embeddings
import tqdm
import os
import re


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


def compute_distances(dct, doc_dct, topdocs, queries, output):
    print("Writing computed distances to "+output+"...")
    default = 0.5
    with tqdm.tqdm(total=os.path.getsize(queries)) as pbar:
        with open(queries, 'r', encoding='utf-8') as f:
            with open(output, 'w', encoding='utf-8') as o:
                for line in f:
                    pbar.update(len(line.encode('utf-8')))
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
                    if not query_term_embeds or len(query_term_embeds) == 0:
                        print("No valid feature vals for query " + query[0] + " (" + query[1] + "), defaulting to " +
                              "cosine distance " + str(default))
                        doc_ids = topdocs.get(query[0])
                        if doc_ids is None or not doc_ids or None in doc_ids:
                            print("Also no top documents for query "+line+", skipping this query!")
                        else:
                            for d in doc_ids:
                                o.write(query[0] + "\t" + d + "\t" + str(default) + '\n')
                    else:
                        doc_ids = topdocs.get(query[0])
                        if doc_ids is None or not doc_ids or None in doc_ids:
                            print("No top documents for query "+line)
                        else:
                            doc_embs = []
                            for d in doc_ids:
                                doc_embs.append(doc_dct.get(d))

                            try:
                                distances = spatial.distance.cdist(np.array(doc_embs), np.array(query_term_embeds), 'cosine').mean(1)
                                for i in range(0, len(doc_embs)):
                                    o.write(query[0] + "\t" + doc_ids[i] + "\t" + str(distances[i]) + '\n')
                            except ValueError as ve:
                                print(ve)
                                if "XA" in ve:
                                    print(format(doc_embs))
                                elif "XB" in ve:
                                    print("Something went wrong with query "+line)
                                    print(format(query_term_embeds))


if __name__ == "__main__":
    top100 = load_top_100("/home/lucile/Documents/nlp/train/msmarco-doctrain-top100")
    qs = "/home/lucile/Documents/nlp/train/msmarco-doctrain-queries.tsv"

    glove_dict = '/home/lucile/Documents/nlp/wordvectors/glove.840B.300d.txt'
    glove_docs = "/home/lucile/Documents/nlp/corpus/msmarco-docs-embeddings-glove.txt"
    dictionary = load_embeddings.load(glove_dict, 300)
    docs_dictionary = load_docs(glove_docs)
    compute_distances(dictionary, docs_dictionary, top100, qs, "/home/lucile/Documents/nlp/train/train-glove-features.tsv")

    del dictionary
    del docs_dictionary

    ft_dict = '/home/lucile/Documents/nlp/wordvectors/crawl-300d-2M.vec'
    ft_docs = "/home/lucile/Documents/nlp/corpus/msmarco-docs-embeddings-fasttext.txt"
    dictionary = load_embeddings.load(ft_dict, 300)
    docs_dictionary = load_docs(ft_docs)
    compute_distances(dictionary, docs_dictionary, top100, qs, "/home/lucile/Documents/nlp/train/train-fasttext-features.tsv")
