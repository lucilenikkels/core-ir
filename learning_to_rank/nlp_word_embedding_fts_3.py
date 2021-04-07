import nlp_load_embeddings as load_embeddings
import numpy as np
from scipy import spatial
import re
import os
import tqdm

NUM_DOCS = 3213835
BOUNDS = [0, 321383, 642767, 964150, 1285534, 1606917, 1928301, 2249684, 2571068, 2892451, 3213835]


def load_queries(filename, dct):
    print("Loading query embeddings " + filename + "...")
    querys = []
    querynames = []
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            query = line.split('\t')
            query_tokens = query[1].replace('/', ' ').replace('.', ' ').split()
            query_term_embeds = []
            for token in query_tokens:
                embed = dct.get(token)
                if embed is not None and np.isfinite(embed).all():
                    query_term_embeds.append(embed)
                else:
                    alt = dct.get(re.sub("[^0-9a-zA-Z]+", "", token))
                    if alt is not None and np.isfinite(alt).all():
                        query_term_embeds.append(alt)
            querys.append(query_term_embeds)
            querynames.append(query[0])
    print("Loaded")
    return querynames, querys


def compute_distances(qids, queries, doc_file, output):
    default = 0.5
    if os.path.exists(output):
        append_write = 'a'  # append if already exists
        print("Appending query-document features to file " + output)
    else:
        append_write = 'w'  # make a new file if not
        print("Writing query-document features to file " + output)

    with tqdm.tqdm(total=os.path.getsize(doc_file)) as pbar:
        with open(doc_file, 'r', encoding='utf-8') as f:
            with open(output, append_write, encoding='utf-8') as o:
                for line in f:
                    pbar.update(len(line.encode('utf-8')))
                    values = line.split(';')
                    doc_emb = np.asarray(values[1].split(), dtype=np.float32)

                    for i in range(len(qids)):
                        if not queries[i]:
                            print(
                                "No valid feature vals for query " + qids[i] + ", defaulting to " +
                                "cosine distance " + str(default))
                            o.write(qids[i] + "\t" + values[0] + "\t" + str(default) + '\n')
                        else:
                            distance = spatial.distance.cdist(queries[i], np.array(doc_emb), 'cosine').mean()
                            o.write(qids[i] + "\t" + values[0] + "\t" + str(distance) + '\n')


if __name__ == "__main__":
    doc_embeddings = [{'name': "glove", 'loc': "/home/lucile/Documents/nlp/corpus/msmarco-docs-embeddings-glove.txt",
                       'emb_loc': '/home/lucile/Documents/nlp/wordvectors/glove.840B.300d.txt',
                       'dims': 300},
                      {'name': "fasttext",
                       'loc': "/home/lucile/Documents/nlp/corpus/msmarco-docs-embeddings-fasttext.txt",
                       'emb_loc': '/home/lucile/Documents/nlp/wordvectors/crawl-300d-2M.vec',
                       'dims': 300}]

    qs = [  # "/home/lucile/Documents/nlp/test/msmarco-test2019-queries.tsv",
        "/home/lucile/Documents/nlp/val/msmarco-docdev-queries.tsv"]
    # "/home/lucile/Documents/nlp/train/msmarco-doctrain-queries.tsv"]

    for tup in doc_embeddings:
        dictionary = load_embeddings.load(tup['emb_loc'], tup['dims'])
        for q_file in qs:
            ids, ques = load_queries(q_file, dictionary)
            compute_distances(ids, ques, tup['loc'], q_file[0:len(q_file) - 7] + "y-doc-fts-" + tup['name'] + ".tsv")
