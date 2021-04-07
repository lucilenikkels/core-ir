import nlp_load_embeddings as load_embeddings
import numpy as np
from scipy import spatial
import re
import faiss


NUM_DOCS = 3213835
DIMS = 300
BOUNDS = [0, 321383, 642767, 964150, 1285534, 1606917, 1928301, 2249684, 2571068, 2892451, 3213835]


def load_docs(filename):
    print("Loading document embeddings " + filename + "...")
    counter = 0
    embeddings = []
    docids = []
    batch = 1
    with open(filename, 'r', encoding="utf-8") as f:
        for line in f:
            if BOUNDS[batch - 1] <= counter < BOUNDS[batch]:
                values = line.split(';')
                vector = np.asarray(values[1].split(), dtype=np.float32)
                embeddings.append(vector.tolist())
                docids.append(values[0])
            elif counter >= BOUNDS[batch]:
                break
            counter += 1
    print("Loaded.")
    print("Building index")
    vectors = np.array(embeddings)
    del embeddings
    index = faiss.index_factory(DIMS, 'IVF100,Flat', faiss.METRIC_INNER_PRODUCT)
    res = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(res, 0, index)
    print("Normalizing vectors")
    faiss.normalize_L2(vectors)
    #gpu_index.train(vectors)
    print("Filling GPU index")
    gpu_index.add(vectors)
    return docids, gpu_index


def compute_distances(dct, docs, index, queryfile, output):
    print("Writing query-document features to file " + output)
    default = 0.5
    with open(queryfile, 'r', encoding='utf-8') as f:
        for line in f:
            query = line.split('\t')
            print(query[0])
            query_tokens = query[1].replace('/', ' ').replace('.', ' ').split()
            query_term_embeds = []
            for token in query_tokens:
                embed = dct.get(token)
                if embed is not None and np.isfinite(embed).all():
                    embed
                    query_term_embeds.append(embed)
                else:
                    alt = dct.get(re.sub("[^0-9a-zA-Z]+", "", token))
                    if alt is not None and np.isfinite(alt).all():
                        query_term_embeds.append(alt)
            if not query_term_embeds:
                print("No valid feature vals for query " + query[0] + " (" + query[1] + ")")
            else:
                print(str(len(query_term_embeds)))
                faiss.normalize_L2(query_term_embeds)
                D, I = index.search(query_term_embeds, index.ntotal)
                print(D)
                break


if __name__ == "__main__":
    doc_embeddings = [{'name': "glove", 'loc': "/home/lucile/Documents/nlp/corpus/msmarco-docs-embeddings-glove.txt",
                       'emb_loc': '/home/lucile/Documents/nlp/wordvectors/glove.840B.300d.txt',
                       'dims': 300},
                      {'name': "fasttext",
                       'loc': "/home/lucile/Documents/nlp/corpus/msmarco-docs-embeddings-fasttext.txt",
                       'emb_loc': '/home/lucile/Documents/nlp/wordvectors/crawl-300d-2M.vec',
                       'dims': 300}]

    queries = [  "/home/lucile/Documents/nlp/test/msmarco-test2019-queries.tsv",
        "/home/lucile/Documents/nlp/val/msmarco-docdev-queries.tsv",
        "/home/lucile/Documents/nlp/train/msmarco-doctrain-queries.tsv"]

    for tup in doc_embeddings:
        dictionary = load_embeddings.load(tup['emb_loc'], tup['dims'])
        doc_ids, index = load_docs(tup['loc'])
        for q_file in queries:
            compute_distances(dictionary, doc_ids, index, q_file,
                              q_file[0:len(q_file) - 7] + "y-doc-fts-" + tup['name'] + ".tsv")
