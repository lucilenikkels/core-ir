import numpy as np
import re
from scipy import spatial


def load_stopwords():
    stopwords = []
    with open('nltk_stopwords.txt', 'r') as r:
        for line in r:
            stopwords.append(line.replace('\n', ''))
    return stopwords


def expand(query_file, output_file, dictionary, k):
    print("Performing expansion for " + output_file)

    stop_words = load_stopwords()
    print("Loaded stopwords: " + ' '.join(stop_words))

    with open(query_file, 'r', encoding="utf-8") as r:
        with open(output_file + '.tsv', 'w') as o:
            for line in r:
                word_tokens = line.split()
                values = [w for w in word_tokens if w not in stop_words]
                rel_terms = []

                embeddings = []
                for word in values[1:]:
                    tmp = dictionary.get(word)
                    cln_word = re.sub("[^0-9a-zA-Z]+", "", word)
                    cln_tmp = dictionary.get(cln_word)

                    if tmp is not None and np.isfinite(tmp).all():
                        embeddings.append(dictionary[word])
                    elif cln_tmp is not None and np.isfinite(cln_tmp).all():
                        embeddings.append(dictionary[cln_word])
                    else:
                        print("No valid dictionary value for " + cln_word + " (" + word + ")")

                if len(embeddings) > 0:
                    emb = np.mean(np.array(embeddings), axis=0)
                    rel_terms = sorted(dictionary.keys(),
                                       key=lambda w: spatial.distance.euclidean(dictionary[w], emb))[0:k]

                res = line.replace(';', ' ').replace('\n', ' ; ')
                for term in rel_terms:
                    res = res + " " + term

                o.write(res + "\n")
    print("Done.")
