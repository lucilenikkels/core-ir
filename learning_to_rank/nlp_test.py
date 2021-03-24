import find_closest
import load_embeddings
import plot_embeddings
import expand_query

GLOVE_FILE = '/home/lucile/Documents/nlp/wordvectors/glove.840B.300d.txt'
FTT_FILE = '/home/lucile/Documents/nlp/wordvectors/crawl-300d-2M.vec'
FILES = [GLOVE_FILE, FTT_FILE]
DIMENSIONS = 300

if __name__ == "__main__":
    #model = load_embeddings.load_w2v_format(GLOVE_FILE, DIMENSIONS)

    # Prints the top 5 most related words to 'king'.
    # The first word can be discarded because it is always going to be 'king' itself.
    #print(model.get_vector("king"))
    #result = model.most_similar(positive=['woman', 'king'], negative=['man'], topn=5)
    #print(result)

    # Plot the N-dimensional vectors in 2-dimensional space
    # plot_embeddings.show(embeddings)

    queries = ["/home/lucile/Documents/nlp/test/msmarco-test2019-queries.tsv",
               "/home/lucile/Documents/nlp/val/msmarco-docdev-queries.tsv",
               "/home/lucile/Documents/nlp/train/msmarco-doctrain-queries.tsv"]
    queries_exp = [["/home/lucile/Documents/nlp/test/msmarco-test2019-queries-exp-glove",
                    "/home/lucile/Documents/nlp/val/msmarco-docdev-queries-exp-glove.",
                    "/home/lucile/Documents/nlp/train/msmarco-doctrain-queries-exp-glove"],
                   ["/home/lucile/Documents/nlp/test/msmarco-test2019-queries-exp-ftt",
                    "/home/lucile/Documents/nlp/val/msmarco-docdev-queries-exp-ftt",
                    "/home/lucile/Documents/nlp/train/msmarco-doctrain-queries-exp-ftt"]]

    k = [5, 10, 25, 50]

    for f in range(0, len(FILES)):
        embeddings = load_embeddings.load(FILES[f], DIMENSIONS)
        for i in range(0, len(queries)):
            expand_query.expand(queries[i], queries_exp[f][i], embeddings, k)
