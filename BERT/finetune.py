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
from collections import defaultdict

# from torch.cuda.amp import autocast
# from torch.cuda.amp import GradScaler
# scaler = GradScaler()

# process(DistilBertModel, DistilBertTokenizer, "distilbert-base-uncased")
# process(AlbertModel, AlbertTokenizer, "albert-base-v1")
# process(AlbertModel, AlbertTokenizer, "albert-xxlarge-v2")

model_class = DistilBertForSequenceClassification
tokenizer_class = DistilBertTokenizerFast
model_name = "distilbert-base-uncased"
field_name = 'distilbert'

writer = SummaryWriter(log_dir='./logs/', comment=model_name)
tokenizer = tokenizer_class.from_pretrained(model_name)

def load_tokens(path, separator):

    # check if a pickle exists for this path (this is purely a performance speed up)
    pck_path = path + ".pkl"
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
            tokens = [int(x) for x in tokens]
            arr = np.array(tokens, dtype='int32')
            dict[id] = arr
    print("Loaded " + path + " into memory size [" + str(len(dict)) + "].")

    print("Building pickle object for faster loading next time...")
    with open(pck_path, 'wb') as output:
        pickle.dump(dict, output, pickle.HIGHEST_PROTOCOL)

    return dict

def load_relevance(path):
    relevance_pairs = set()
    with open(path) as infile:
        for line in infile:
            # print("line", line)
            split = line.split(" ")
            query_id = split[0]
            doc_id = split[2]
            relevance = int(split[3])
            if relevance != 1:
                print("Expected relevance 1 but found", relevance, "skipping...")
                continue
            relevance_pairs.add((query_id, doc_id))
    print("Loaded relevance " + path + " into memory size [" + str(len(relevance_pairs)) + "].")
    return relevance_pairs

# load document tokens into memory
doc_tokens = load_tokens("doc_tokens_" + model_name + ".txt", separator=" ")
# load query tokens into memory
query_tokens = load_tokens("./train/query_tokens_" + model_name + ".txt", separator=" ")
# load the relevance judgements
relevance = load_relevance('./train/msmarco-doctrain-qrels.tsv')

def load_training_pairs(path):
    query_dict = {}

    with open(path) as infile:
        for line in infile:
            split = line.split(" ")
            qid = split[0]
            if qid not in query_dict:
                query_dict[qid] = list()

            did = split[2]
            query_dict[qid].append(did)
    
    dataset = []

    # now we build the training data from this
    qids = query_dict.keys()
    for qid in qids:
        positive = []
        negative = []

        top100_did = query_dict[qid]
        for did in top100_did:
            # construct a pair for lookup into relevance set
            pair = (qid, did)
            if pair in relevance:
                positive.append(pair)
            else:
                negative.append(pair)

        random.shuffle(negative)
        random.shuffle(positive)
        # randomly subsample to balance the dataset more
        negative = negative[0:1] # or 1:10 as per https://arxiv.org/pdf/2009.09392.pdf
        # print("set sizes", len(negative), len(positive))

        if len(positive) == 0:
            continue

        for sample in negative:
            dataset.append((sample, 0.0))
        for sample in positive:
            dataset.append((sample, 1.0))
    
    random.shuffle(dataset)
    print(20 * "-")
    print("Total dataset size", len(dataset))
    print(20 * "-")
    return dataset # [0:32] * 10000

training_pairs = load_training_pairs('./train/msmarco-doctrain-top100')

# quick list for access to all known keys and queries
all_docs = list(doc_tokens.keys())
all_queries = list(query_tokens.keys())

def load_pair_data(query_id, doc_id):
    # lookup the tokens
    tokens_q = query_tokens[query_id]
    tokens_d = doc_tokens[doc_id]

    if len(tokens_q) == 0 or len(tokens_d) == 0:
        raise Exception("Token length was 0")

    # convert back into actual words
    tokens_q = tokenizer.convert_ids_to_tokens(tokens_q)
    tokens_d = tokenizer.convert_ids_to_tokens(tokens_d)
    # print('QUERY', tokens_q, "DOCUMENT_cut", tokens_d[0:16])
    return tokens_q, tokens_d

class RelevanceDataset(data.Dataset):

    def __len__(self):
        return len(training_pairs)

    def __getitem__(self, index):
        pair, label = training_pairs[index]
        return (load_pair_data(*pair), label)


EPOCHS = 1
BATCH_SIZE = 32
ACCUM_STEPS = 8
STEPS_PER_EPOCH = len(training_pairs) // BATCH_SIZE
STEPS = STEPS_PER_EPOCH * EPOCHS
WARMUP_STEPS = 1024

print(20 * "=")
print('EPOCHS', EPOCHS)
print('BATCH_SIZE', BATCH_SIZE)
print('RELEVANCE_SIZE', len(relevance))
print('STEPS_PER_EPOCH', STEPS_PER_EPOCH)
print('STEPS', STEPS)
print('WARMUP_STEPS', WARMUP_STEPS)
print(20 * "=")

# use collate_fn to manually construct the batch
def collate_fn(list):
    # this receives a list of items returned by the dataset
    queries = []
    documents = []
    labels = []

    for item in list:
        # see RelevanceDataset __getitem__ for the returned format
        queries.append(item[0][0])
        documents.append(item[0][1])
        labels.append([item[1]])

    # now combine all the data into a single batch by calling the tokenizer
    batch = tokenizer(text=queries, text_pair=documents, is_split_into_words=True, padding=True,
                        truncation='longest_first', return_tensors='pt', max_length=256, return_attention_mask=True)
                        
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']

    # print(tokenizer.decode(input_ids[0]))
    # exit()

    labels = torch.tensor(labels)
    return input_ids, attention_mask, labels

# set drop_last to True to drop the last batch if it is not full
dataloader = data.DataLoader(RelevanceDataset(), batch_size=BATCH_SIZE, shuffle=True, num_workers=4, collate_fn=collate_fn, drop_last=True)

# we will fune-tune an existing model, so lets start by loading the original model
model = model_class.from_pretrained(model_name, num_labels=1) # 1 extra output for binary classification
model = model.to('cuda')
model.train()

def set_frozen(frozen):
    print("SET WEIGHTS FROZEN TO ", frozen)
    for param in getattr(model, field_name).parameters():
        param.requires_grad = (not frozen)

# set_frozen(True)

"""
# freeze the first encoder layers
for param in model.base_model.parameters():
    param.requires_grad = False
"""

print("Running a total of", STEPS, "steps.")
print("First doing", WARMUP_STEPS, "warmup steps.")


# initialize the recommended optimizer
optimizer = AdamW(model.parameters(), lr=3e-5, correct_bias=False) # , weight_decay=0.01
scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=WARMUP_STEPS // ACCUM_STEPS,
                                                num_training_steps=(STEPS - WARMUP_STEPS) // ACCUM_STEPS)
# scaler = GradScaler() # for fp16 training

step = 0
start_time = time.time()
summed_loss = 0

for epoch in range(EPOCHS):
    print("RUNNING EPOCH", epoch)

    # shuffle the relevance for the next epoch
    random.shuffle(training_pairs)
    

    for input_ids, attention_mask, labels in dataloader:
        step += 1

        # if step == WARMUP_STEPS:
        # we now unfreeze the encoder layers
        # set_frozen(False)

        # print('input_ids shape', input_ids.shape)
        # print('attention_mask shape', attention_mask.shape)
        # print('labels shape', labels.shape)

        # pass this batch through the model
        model_start_time = time.time()
        optimizer.zero_grad()

        # with autocast():
        outputs = model(input_ids.cuda(), attention_mask=attention_mask.cuda())
        out_values = outputs.logits
        loss = F.binary_cross_entropy_with_logits(out_values, labels.cuda())
        summed_loss += float(loss)
        (loss / ACCUM_STEPS).backward() # scaler.scale(
        # scaler.unscale_(optimizer) # needed for clip_grad_norm_

        now = time.time()
        took = int((now - start_time) * 1000)
        start_time = now 

        if step % ACCUM_STEPS == 0:

            # clip the norm of the gradients to 1.0, gradient clipping is not in AdamW anymore
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step() # scaler.step(
            scheduler.step()
            # scaler.update()

            if step % 32 == 0:
                print("Running step", step, "/", STEPS, "LOSS", float(loss), 'took ms', took, 'of which was cuda ms', int((now - model_start_time) * 1000))
                print(labels.flatten()[0:16])
                print([str(x)[0:4] for x in torch.sigmoid(out_values).flatten()[0:16].detach().cpu().numpy()])

            # also write average to graph
            writer.add_scalar('loss', summed_loss / ACCUM_STEPS, step)
            writer.flush()
            summed_loss = 0

        if step % (1024 * 8) == 0:
            model.save_pretrained("latest_" + model_name + ".model")


# save the final model
model.save_pretrained("latest_" + model_name + ".model")