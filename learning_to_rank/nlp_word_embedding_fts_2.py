import nlp_load_embeddings as load_embeddings
import numpy as np
from scipy import spatial
import re


def load_docs(filename):
    print("Loading document embeddings " + filename + "...")
    embeddings_dict = {}
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split(';')
            vector = np.asarray(values[1].split(), dtype=np.float32)
            embeddings_dict[values[0]] = vector
    print("Loaded.")
    return embeddings_dict


def compute_distances(dct, documents, queryfile, output):
    print("Writing query-document features to file " + output)
    default = 0.5
    with open(queryfile, 'r', encoding='utf-8') as f:
        with open(output, 'w', encoding='utf-8') as o:
            for line in f:
                query = line.split('\t')
                print(query[0])
                query_tokens = query[1].replace('/', ' ').replace('.', ' ').split()
                query_term_embeds = []
                for token in query_tokens:
                    embed = dct.get(token)
                    alt = dct.get(re.sub("[^0-9a-zA-Z]+", "", token))
                    if embed is not None and np.isfinite(embed).all():
                        query_term_embeds.append(embed)
                    elif alt is not None and np.isfinite(alt).all():
                        query_term_embeds.append(alt)
                for doc_id in documents:
                    dists = []
                    for e in query_term_embeds:
                        dists.append(spatial.distance.cosine(documents[doc_id],  e))
                    if not dists:
                        print("No valid feature for query " + query[0] + " (" + query[1] + ") and document " + doc_id
                              + ", defaulting to cosine distance " + str(default))
                        o.write(query[0] + "\t" + doc_id + "\t" + str(default) + '\n')
                    else:
                        o.write(query[0] + "\t" + doc_id + "\t" + str(np.mean(dists)) + '\n')


if __name__ == "__main__":
    doc_embeddings = [{'name': "glove", 'loc': "/home/lucile/Documents/nlp/corpus/msmarco-docs-embeddings-glove.txt",
                       'emb_loc': '/home/lucile/Documents/nlp/wordvectors/glove.840B.300d.txt',
                       'dims': 300},
                      {'name': "fasttext", 'loc': "/home/lucile/Documents/nlp/corpus/msmarco-docs-embeddings-fasttext.txt",
                       'emb_loc': '/home/lucile/Documents/nlp/wordvectors/crawl-300d-2M.vec',
                       'dims': 300}]

    queries = ["/home/lucile/Documents/nlp/test/msmarco-test2019-queries.tsv",
               "/home/lucile/Documents/nlp/val/msmarco-docdev-queries.tsv",
               "/home/lucile/Documents/nlp/train/msmarco-doctrain-queries.tsv"]

    for tup in doc_embeddings:
        dictionary = load_embeddings.load(tup['emb_loc'], tup['dims'])
        docs_dict = load_docs(tup['loc'])
        for q_file in queries:
            compute_distances(dictionary, docs_dict, q_file, q_file[0:len(q_file)-7]+"y-doc-fts-"+tup['name']+".tsv")
