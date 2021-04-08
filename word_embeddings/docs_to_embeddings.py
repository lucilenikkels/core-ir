import numpy as np

DOCS = "/home/lucile/Documents/nlp/corpus/msmarco-docs.trec"


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


def parse_doc(dct, doc_text, doc_id):
    tokens = doc_text.split()
    embeddings = []
    for token in tokens:
        tmp = dct.get(token)
        if tmp is not None and np.isfinite(tmp).all():
            embeddings.append(tmp)
    if not embeddings:
        tokens = doc_text.replace('/', ' ').replace('.', ' ').split()
        for token in tokens:
            tmp = dct.get(token)
            if tmp is not None and np.isfinite(tmp).all():
                embeddings.append(tmp)
        if not embeddings:
            print("No embeddings found for document " + str(doc_id))
            return np.array([])
    return np.mean(np.array(embeddings), axis=0)


def parse_docs_with_dict(dct, output):
    print("Writing document embeddings to file " + output)
    with open(DOCS, 'r', encoding='utf-8') as f:
        with open(output, 'w', encoding='utf-8') as o:
            for line in f:
                ind = line.find("</DOCNO>")
                if ind != -1:
                    cur_id = line[len("<DOCNO>"):ind]
                elif line.find("</DOC>") != -1:
                    res = parse_doc(dct, cur_text, cur_id)
                    o.write(cur_id + ";" + " ".join(str(v) for v in res) + '\n')
                elif line.find('<DOC>') != -1:
                    cur_id = "0"
                    cur_text = ""
                elif line.find('TEXT>') == -1:
                    cur_text = cur_text + line


if __name__ == "__main__":
    embedding_types = [{'loc': '/home/lucile/Documents/nlp/wordvectors/glove.840B.300d.txt',
                        'output': "/home/lucile/Documents/nlp/corpus/msmarco-docs-embeddings-glove.txt",
                        'dims': 300},
                       {'loc': '/home/lucile/Documents/nlp/wordvectors/crawl-300d-2M.vec',
                        'output': "/home/lucile/Documents/nlp/corpus/msmarco-docs-embeddings-fasttext.txt",
                        'dims': 300}]

    for tup in embedding_types:
        dictionary = load_embeddings(tup['loc'], tup['dims'])
        parse_docs_with_dict(dictionary, tup['output'])
