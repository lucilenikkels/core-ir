import tqdm
import os


if __name__ == "__main__":
    top100file = "/home/lucile/Documents/nlp/val/msmarco-docdev-top100"
    output = "/home/lucile/Documents/nlp/val/dev-glove-features.tsv"
    alldocs = "/home/lucile/Documents/nlp/val/msmarco-docdev-query-doc-fts-glove.tsv"

    dictionary = {}
    with open(top100file, 'r', encoding='utf-8') as f:
        for line in f:
            vals = line.split()
            if vals[0] in dictionary:
                dictionary[vals[0]].append(vals[2])
            else:
                dictionary[vals[0]] = [vals[2]]
    print("Dictionary loaded")

    with tqdm.tqdm(total=os.path.getsize(alldocs)) as pbar:
        with open(alldocs, 'r', encoding='utf-8') as f:
            with open(output, 'w', encoding='utf-8') as o:
                for line in f:
                    pbar.update(len(line.encode('utf-8')))
                    vals = line.split('\t')
                    if vals[1] in dictionary.get(vals[0]):
                        o.write(line.replace('\n', '')+'\n')
