import numpy as np
import re
from scipy import spatial


def expand(query_file, output_file, dictionary, k):
    print("Performing expansion for "+output_file)

    num_exps = {}
    for i in range(0, len(k)):
        num_exps[str(k[i])] = output_file + "-" + str(k[i]) + ".tsv"

    with open(query_file, 'r', encoding="utf-8") as r:
        for line in r:
            values = line.split()
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
                    print("No valid dictionary value for " + re.sub("[^0-9a-zA-Z]+", "", word) + " (" + word + ")")

            if len(embeddings) > 0:
                emb = np.mean(np.array(embeddings), axis=0)
                rel_terms = sorted(dictionary.keys(),
                                   key=lambda w: spatial.distance.euclidean(dictionary[w], emb))[0:int(list(num_exps)[-1])]

            for num in num_exps:
                sublst = rel_terms[0:int(num)]
                res = line.replace('\n', ' , ')
                for term in sublst:
                    if not bool(re.match('^[a-zA-Z0-9]+$', '123#$%abc')):
                        res = res + " " + term

                with open(num_exps[num], 'a') as o:
                    o.write(res + "\n")
    print("Done.")
