import numpy as np
from scipy import spatial


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


def load_top_100(filename, qid):
    print("Loading top 100 documents " + filename + "...")
    lst = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            vals = line.split()
            if vals[0] == qid:
                lst.append(vals[2])
    print("Top 100 list loaded")
    return lst


def compute_distances(dct, doc_dct, q):
    query_embeddings = []
    for term in q["text"].split():
        query_embeddings.append(dct.get(term))

    doc_embeddings = []
    doc_ids = []
    for key in doc_dct:
        doc_embeddings.append(doc_dct[key])
        doc_ids.append(key)

    distances = spatial.distance.cdist(np.array(doc_embeddings), np.array(query_embeddings), 'cosine')

    res = {}
    for i in range(0, len(doc_embeddings)):
        print(doc_ids[i], distances[i], distances[i].mean(), np.sum(distances[i]))
        res[doc_ids[i]] = str(distances[i])

    return res


if __name__ == "__main__":
    top100 = "C:/Users/luusn/Documents/msmarco-doctest2019-top100"
    glove_docs = "C:/Users/luusn/Documents/glove-docs-405717.txt"
    glove_embs = 'C:/Users/luusn/Documents/glove.840B.300d.txt'
    dimensions = 300

    query = {"id": "405717", "text": "is cdg airport in main paris"}

    docs_dictionary = load_docs(glove_docs)
    dictionary = load_embeddings(glove_embs, dimensions)

    compute_distances(dictionary, docs_dictionary, query)
    print("___________________________________________________________")
    query2 = {"id": "405717", "text": "is cdg the airport in main paris"}
    compute_distances(dictionary, docs_dictionary, query2)
    print("___________________________________________________________")
    query2 = {"id": "405717", "text": "cdg airport main paris"}
    compute_distances(dictionary, docs_dictionary, query2)
