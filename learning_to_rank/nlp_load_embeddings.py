import numpy as np
from gensim.models import KeyedVectors


def load(filename, dimensions):
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
            embeddings_dict[word] = vector
    print("Loaded.")
    return embeddings_dict


def load_w2v_format(filename, dimensions):
    print("Loading pre-trained embeddings " + filename + "...")
    model = KeyedVectors.load_word2vec_format(filename, binary=False)
    print("Loaded")
    return model
