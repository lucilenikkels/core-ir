import nlp_load_embeddings as load_embeddings
import numpy as np
from scipy import spatial
import re
import os
import tqdm

NUM_DOCS = 3213835
BOUNDS = [0, 321383, 642767, 964150, 1285534, 1606917, 1928301, 2249684, 2571068, 2892451, 3213835]


def load_docs(filename, b):
    print("Loading document embeddings " + filename + "...")
    counter = 0
    embeddings = []
    docids = []
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            if BOUNDS[b - 1] <= counter < BOUNDS[b]:
                values = line.split(';')
                embeddings.append(np.asarray(values[1].split(), dtype=np.float32).tolist())
                docids.append(values[0])
            elif counter >= BOUNDS[b]:
                break
            counter += 1
    print("Loaded")
    return docids, np.array(embeddings)


def compute_distances(dct, docs, doc_embs, queryfile, output):
    default = 0.5
    if os.path.exists(output):
        append_write = 'a'  # append if already exists
        print("Appending query-document features to file " + output)
    else:
        append_write = 'w'  # make a new file if not
        print("Writing query-document features to file " + output)

    with tqdm.tqdm(total=os.path.getsize(queryfile)) as pbar:
        with open(queryfile, 'r', encoding='utf-8') as f:
            with open(output, append_write, encoding='utf-8') as o:
                for line in f:
                    pbar.update(len(line.encode('utf-8')))
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
                    if not query_term_embeds:
                        print("No valid feature vals for query " + query[0] + " (" + query[1] + "), defaulting to " +
                              "cosine distance " + str(default))
                        for d in docs:
                            o.write(query[0] + "\t" + d + "\t" + str(default) + '\n')
                    else:
                        print(dct.get(query_tokens[0]))
                        print(query_term_embeds)
                        distances = spatial.distance.cdist(doc_embs, np.array(query_term_embeds), 'cosine').mean(1)
                        for i in range(0, len(docs)):
                            o.write(query[0] + "\t" + docs[i] + "\t" + str(distances[i]) + '\n')


if __name__ == "__main__":
    doc_embeddings = [#{'name': "glove", 'loc': "/home/lucile/Documents/nlp/corpus/msmarco-docs-embeddings-glove.txt",
                      # 'emb_loc': '/home/lucile/Documents/nlp/wordvectors/glove.840B.300d.txt',
                      # 'dims': 300},
                      {'name': "fasttext",
                       'loc': "/home/lucile/Documents/nlp/corpus/msmarco-docs-embeddings-fasttext.txt",
                       'emb_loc': '/home/lucile/Documents/nlp/wordvectors/crawl-300d-2M.vec',
                       'dims': 300}]

    queries = [  # "/home/lucile/Documents/nlp/test/msmarco-test2019-queries.tsv",
        "/home/lucile/Documents/nlp/val/msmarco-docdev-queries.tsv"]
    # "/home/lucile/Documents/nlp/train/msmarco-doctrain-queries.tsv"]

    for tup in doc_embeddings:
        dictionary = load_embeddings.load(tup['emb_loc'], tup['dims'])
        for batch in range(1, 11):
            print("Processing batch " + str(batch) + "/10")
            doc_ids, docs_mat = load_docs(tup['loc'], batch)
            for q_file in queries:
                compute_distances(dictionary, doc_ids, docs_mat, q_file,
                                  q_file[0:len(q_file) - 7] + "y-doc-fts-" + tup['name'] + ".tsv")
