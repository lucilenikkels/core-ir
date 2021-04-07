## AUTHORS - Sander Gielisse || Lucile Nikkels
import torch
import torch.nn.functional as F
import random
from transformers import BertTokenizer, BertModel, DistilBertTokenizerFast, DistilBertForSequenceClassification, AlbertTokenizer, AlbertModel
from transformers import AdamW
import pickle
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch.utils.data as data
from transformers import get_linear_schedule_with_warmup
import time
import gc
from torch.cuda.amp import autocast

# from torch.cuda.amp import autocast
# from torch.cuda.amp import GradScaler

# process(DistilBertModel, DistilBertTokenizer, "distilbert-base-uncased")
# process(AlbertModel, AlbertTokenizer, "albert-base-v1")
# process(AlbertModel, AlbertTokenizer, "albert-xxlarge-v2")

MAX_LENGTH = 256
model_name = "distilbert-base-uncased"
token_path = "./test/query_tokens_" + model_name + ".txt"
input_features = './test/filtered_test_features.txt'
model_class = DistilBertForSequenceClassification
tokenizer_class = DistilBertTokenizerFast
tokenizer = tokenizer_class.from_pretrained(model_name)

def load_tokens(path, separator):

    # check if a pickle exists for this path (this is purely a performance speed up)
    pck_path = path + "_len" + str(MAX_LENGTH) + ".pkl"
    if os.path.exists(pck_path):
        print("Existing pickle found at ", pck_path, " loading...")
        with open(pck_path, 'rb') as p:
            return pickle.load(p)

    print("Loading " + path + " into memory...")
    dict = {}
    counter = 0
    with open(path) as infile:
        for line in infile:
            if counter % 10000 == 0:
                print("Progress", counter)
            counter += 1

            split = line.split(separator)
            id = split[0]
            tokens = split[1].split(",")
            tokens = [int(x) for x in tokens][:MAX_LENGTH]
            arr = np.array(tokens, dtype='int32')
            dict[id] = arr
    print("Loaded " + path + " into memory size [" + str(len(dict)) + "].")

    print("Building pickle object for faster loading next time...")
    with open(pck_path, 'wb') as output:
        pickle.dump(dict, output, pickle.HIGHEST_PROTOCOL)

    return dict

# these are in format
# 2 qid:156493 1:-4.69200 2:22.35779 ... 15:0.43943 16:0.78389 #D683584
def load_feature_pairs(path):

    # check if a pickle exists for this path (this is purely a performance speed up)
    pck_path = path + ".pkl"
    if os.path.exists(pck_path):
        print("Existing pickle found at ", pck_path, " loading...")
        with open(pck_path, 'rb') as p:
            return pickle.load(p)

    relevance_pairs = []
    counter = 0
    with open(path) as infile:
        for line in infile:

            if counter % 100000 == 0:
                print("Progress", counter)
            counter += 1

            line = line.strip('\n')
            # print("line", line)
            split = line.split(" ")
            target = split[0]
            query_id = split[1].split(":")[1] # format qid:156493
            features = ' '.join(split[2:-1])
            doc_id = split[-1][1:] # format #D683584
            pair = (target, query_id, doc_id, features)
            # print(pair)
            relevance_pairs.append(pair)
    print("Loaded feature pairs " + path + " into memory size [" + str(len(relevance_pairs)) + "].")

    print("Building pickle object for faster loading next time...")
    with open(pck_path, 'wb') as output:
        pickle.dump(relevance_pairs, output, pickle.HIGHEST_PROTOCOL)

    return relevance_pairs


# load document tokens into memory
doc_tokens = load_tokens("doc_tokens_" + model_name + ".txt", separator=" ")
# load query tokens into memory
query_tokens = load_tokens(token_path, separator=" ")

# load the pairs  judgements
feature_pairs = load_feature_pairs(input_features)


def load_pair_data(query_id, doc_id):
    # lookup the tokens
    tokens_q = query_tokens[query_id]
    tokens_d = doc_tokens[doc_id]
    # convert back into actual words
    tokens_q = tokenizer.convert_ids_to_tokens(tokens_q)
    tokens_d = tokenizer.convert_ids_to_tokens(tokens_d)
    # print('QUERY', tokens_q, "DOCUMENT_cut", tokens_d[0:16])
    return tokens_q, tokens_d

class RelevanceDataset(data.Dataset):

    def __len__(self):
        return len(feature_pairs)

    def __getitem__(self, index):
        target, query_id, doc_id, features = feature_pairs[index] # (target, query_id, doc_id, features)
        tokens_q, tokens_d = load_pair_data(query_id, doc_id)
        return target, query_id, doc_id, features, tokens_q, tokens_d

BATCH_SIZE = 512
print('BATCH_SIZE', BATCH_SIZE)

# use collate_fn to manually construct the batch
def collate_fn(list):
    # this receives a list of items returned by the dataset
    targets = []
    query_ids = []
    doc_ids = []
    features_list = []

    queries = []
    documents = []

    for target, query_id, doc_id, features, tokens_q, tokens_d in list:
        
        targets.append(target)
        query_ids.append(query_id)
        doc_ids.append(doc_id)
        features_list.append(features)

        queries.append(tokens_q)
        documents.append(tokens_d)

    # now combine all the data into a single batch by calling the tokenizer
    batch = tokenizer(text=queries, text_pair=documents, is_split_into_words=True, padding=True,
                        truncation='longest_first', return_tensors='pt', max_length=MAX_LENGTH, return_attention_mask=True)
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    
    return input_ids, attention_mask, targets, query_ids, doc_ids, features_list

# the file to write the features to
result_features = open(input_features[:-4] + "_with_" + model_name + ".txt", "w")

# check the resume file _resume
resume_path = input_features[:-4] + "_with_" + model_name + "_resume.txt"
if os.path.exists(resume_path):
    print("Resuming generation from previously unfinished file...")
    already_done = set()
    with open(resume_path, "r") as resume_file:
        for done_line in resume_file:
            if done_line.endswith("\n"): # then it is a full line
                # parse the (qid,did) pair and write the line to the result features
                # 2 qid:156493 1:-4.69200 2:22.35779 ... 15:0.43943 16:0.78389 #D683584\n
                split = done_line.split(" ")
                query_id = split[1].split(":")[1] # format qid:156493
                doc_id = split[-1][1:].strip('\n') # format #D683584\n
                already_done.add((query_id, doc_id))
                result_features.write(done_line)

    print("Found", len(already_done), "pairs to already be finished, removing them from the queue...")
    # note pairs are waiting in feature_pairs with format (target, query_id, doc_id, features)
    # now all the already_done pairs have to be remove from feature_pairs
    result_list = []
    for pair in feature_pairs:
        _, query_id, doc_id, _ = pair
        small_pair = (query_id, doc_id)
        if small_pair not in already_done:
            result_list.append(pair)
    feature_pairs = result_list # replace with the new selection
    print("Finished loading from previously unfinished file, ", len(feature_pairs), "feature pairs left.")

# set drop_last to True to drop the last batch if it is not full
dataloader = data.DataLoader(RelevanceDataset(), batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=2, collate_fn=collate_fn, drop_last=False)

# we will fune-tune an existing model, so lets start by loading the original model
model = model_class.from_pretrained("./latest_" + model_name + ".model/", num_labels=1) # 1 extra output for binary classification
cuda = torch.cuda.is_available()
if cuda:
    model = model.to('cuda')
model.eval()

start_time = time.time()
batch_counter = 0
for input_ids, attention_mask, targets, query_ids, doc_ids, features_list in dataloader:

    if cuda:
        input_ids = input_ids.cuda()
        attention_mask = attention_mask.cuda()

    batch_counter += 1

    with torch.no_grad():
        with autocast():
            outputs = model(input_ids, attention_mask=attention_mask)
            out_values = torch.sigmoid(outputs.logits)
        
    # write to the output file in format
    # 2 qid:156493 1:-4.69200 2:22.35779 ... 15:0.43943 16:0.78389 #D683584
    for i in range(input_ids.shape[0]):
        features = features_list[i].split(' ')
        new_feature = float(out_values[i])
        # append it to existing features
        features.append(str(len(features) + 1) + ':' + str(new_feature)[:8])

        target = targets[i]
        query_id = query_ids[i]
        doc_id = doc_ids[i]
        line = target + " qid:" + query_id + " " + (' '.join(features)) + ' #' + doc_id
        result_features.write(line + "\n")
        # result_features.flush()

    now = time.time()
    took = int((now - start_time) * 1000)
    start_time = now 
    print("Running step", batch_counter, "/", len(dataloader), 'took ms', took)

result_features.close()