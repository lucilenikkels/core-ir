import numpy as np

from transformers import BertTokenizerFast, BertModel, DistilBertTokenizerFast, DistilBertModel, AlbertTokenizerFast, AlbertModel

# process(DistilBertModel, DistilBertTokenizer, "distilbert-base-uncased")
# process(AlbertModel, AlbertTokenizer, "albert-base-v1")
# process(AlbertModel, AlbertTokenizer, "albert-xxlarge-v2")

tokenizer_class = DistilBertTokenizerFast
model_name = "distilbert-base-uncased"
tokenizer = tokenizer_class.from_pretrained(model_name)
results = open("./test/query_tokens_" + model_name + ".txt", "w")

counter = 0
with open("./test/msmarco-test2019-queries.tsv") as infile:
    for line in infile:
        line = line.strip('\n')
    
        # generate embedding and save
        split = line.split('\t')
        query_id = split[0]
        query_text = split[1]
        

        tokens = tokenizer.tokenize(query_text)
        to_save = tokens[0:512]
        
        # print('tokens', to_save)
        ids = tokenizer.convert_tokens_to_ids(to_save)
        # print('ids', ids)

        data = ','.join(str(x) for x in ids)
        results.write(query_id + " " + data + "\n")
        results.flush()

        counter += 1
        print("DONE", query_id, "counter", counter, "length", len(query_text))


results.close()