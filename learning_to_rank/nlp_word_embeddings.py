import nlp_load_embeddings as load_embeddings
import nlp_expand_query as expand_query

GLOVE_FILE = '/home/lucile/Documents/nlp/wordvectors/glove.840B.300d.txt'
FTT_FILE = '/home/lucile/Documents/nlp/wordvectors/crawl-300d-2M.vec'
QUERIES = ["/home/lucile/Documents/nlp/test/msmarco-test2019-queries.tsv",
           "/home/lucile/Documents/nlp/val/msmarco-docdev-queries.tsv",
           "/home/lucile/Documents/nlp/train/msmarco-doctrain-queries.tsv"]
QUERIES_EXP = [["/home/lucile/Documents/nlp/test/msmarco-test2019-queries-exp-glove-2",
                "/home/lucile/Documents/nlp/val/msmarco-docdev-queries-exp-glove-2.",
                "/home/lucile/Documents/nlp/train/msmarco-doctrain-queries-exp-glove-2"],
               ["/home/lucile/Documents/nlp/test/msmarco-test2019-queries-exp-ftt-2",
                "/home/lucile/Documents/nlp/val/msmarco-docdev-queries-exp-ftt-2",
                "/home/lucile/Documents/nlp/train/msmarco-doctrain-queries-exp-ftt-2"]]
DIMENSIONS = 300

if __name__ == "__main__":

    # k = [5, 10, 25, 50]
    k = 25
    files = [GLOVE_FILE, FTT_FILE]

    for f in range(0, len(files)):
        embeddings = load_embeddings.load(files[f], DIMENSIONS)
        for i in range(0, len(QUERIES)):
            expand_query.expand(QUERIES[i], QUERIES_EXP[f][i], embeddings, k)
