import numpy as np

from transformers import BertTokenizerFast, BertModel, DistilBertTokenizerFast, DistilBertModel, AlbertTokenizerFast, AlbertModel

# process(DistilBertModel, DistilBertTokenizer, "distilbert-base-uncased")
# process(AlbertModel, AlbertTokenizer, "albert-base-v1")
# process(AlbertModel, AlbertTokenizer, "albert-xxlarge-v2")

tokenizer_class = DistilBertTokenizerFast
model_name = "distilbert-base-uncased"
tokenizer = tokenizer_class.from_pretrained(model_name)
results = open("doc_tokens_" + model_name + ".txt", "w")

counter = 0
with open("./msmarco-docs.trec") as infile:
    for line in infile:
        line = line.strip('\n')
        # print("LINE [", line, "]")
        if line == "<DOC>":
            doc_texts = []
            doc_id = ""
            reading_text = False

        elif line.startswith("<DOCNO>"):
            doc_id = line.replace("<DOCNO>", "").replace("</DOCNO>", "")

        elif line == "<TEXT>":
            reading_text = True

        elif line == "</TEXT>":
            reading_text = False

        elif line == "</DOC>":
            # generate embedding and save
            doc_text = ' '.join(doc_texts)
            tokens = tokenizer.tokenize(doc_text)
            to_save = tokens[0:512]
            
            # print('tokens', to_save)
            ids = tokenizer.convert_tokens_to_ids(to_save)
            # print('ids', ids)

            data = ','.join(str(x) for x in ids)
            results.write(doc_id + " " + data + "\n")
            results.flush()

            counter += 1
            print("DONE", doc_id, "counter", counter, "length", len(doc_text))

        else:
            if not reading_text:
                raise Exception("Encountered line but not reading text [" + line + "]")
            doc_texts.append(line)

results.close()