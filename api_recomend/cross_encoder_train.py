import torch

from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, LoggingHandler, util, models, evaluation, losses, InputExample
import logging
from datetime import datetime
import gzip
import os
import tarfile
from collections import defaultdict
from torch.utils.data import IterableDataset

import json
from sentence_transformers import SentenceTransformer, util
import time
import torch

from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers.cross_encoder.evaluation import CEBinaryClassificationEvaluator



import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--do_train", action="store_true")
parser.add_argument("--do_eval", action="store_true")
parser.add_argument("--eval_no_train", action="store_true")
parser.add_argument("--trained_cross_encoder", type=str)
args = parser.parse_args()



logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout
train_batch_size = 64 #Increasing the train batch size improves the model performance, but requires more GPU memory

#model_name = "SBRT_output/training_biker_bi-encoder-min_5_max_10_ir_10_distilroberta-base_30_iter"
model_name = './SBRT_output/training_biker_bi-encoder-min_5_max_10_ir_10_distilroberta-base-full-best/'
model = SentenceTransformer(model_name)


if args.eval_no_train:
   cross_encoder = CrossEncoder(args.trained_cross_encoder)
else:
   cross_encoder = CrossEncoder('cross-encoder/ms-marco-TinyBERT-L-6')

model_save_path = './SBRT_output/cross-encoder_training_biker-30_iter_TinyBERT'+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

### Now we read the MS Marco dataset
data_folder = '../data/full_data_min_5_max_10_ir_10/'
os.makedirs(data_folder, exist_ok=True)
import json 

corpus = {}

collection_filepath = os.path.join(data_folder, 'Corpus_dict.json')
with open(collection_filepath, 'r', encoding='utf8') as fIn:
  corpus = json.load(fIn)

Passage_dict = {}
collection_filepath = os.path.join(data_folder, 'Passage_dict.json')
with open(collection_filepath, 'r', encoding='utf8') as fIn:
  Passage_dict = json.load(fIn)


evaluate_Corpus = {}
collection_filepath = os.path.join(data_folder, 'evaluate_Corpus.json')
with open(collection_filepath, 'r', encoding='utf8') as fIn:
  evaluate_Corpus = json.load(fIn)

evaluate_queries = {}
collection_filepath = os.path.join(data_folder, 'evaluate_queries.json')
with open(collection_filepath, 'r', encoding='utf8') as fIn:
   evaluate_queries = json.load(fIn)

evaluate_rel_doc = {}
collection_filepath = os.path.join(data_folder, 'evaluate_rel_doc.json')
with open(collection_filepath, 'r', encoding='utf8') as fIn:
   evaluate_rel_doc = json.load(fIn)



def get_triplets(Passage_dict):
    Triplets= []
    for k, v in Passage_dict.items():
        for x in v[0]:
            for y in v[1]:
                # query,same_appi,diff_api
                Triplets.append([k,x,y])
    return Triplets

train_triplets = get_triplets(Passage_dict)


print(len(train_triplets))


print(len(corpus))
# print(len(Answers_dict))
print(len(Passage_dict))
print(len(evaluate_queries))
print(len(evaluate_Corpus))
print(len(evaluate_rel_doc))
#print(len(dev_queries))
print(len(corpus))
#print(len(dev_rel_docs))

train_dataset=[]        
for triplet in train_triplets:
    qid, pos_id, neg_id = triplet
    
    qid = str(qid)
    pos_id = str(pos_id)
    neg_id = str(neg_id)

    query_text = corpus[qid]
    pos_text = corpus[pos_id]
    neg_text = corpus[neg_id]
    #print('query_text:',query_text, 'pos_text:',pos_text,'neg_text:',neg_text)
    pos_instance = InputExample(texts=[query_text, pos_text],label=1)
    neg_instance = InputExample(texts=[query_text, neg_text],label=0)

    train_dataset.append(pos_instance)
    train_dataset.append(neg_instance)
print('len train_dataset',len(train_dataset))


## Read Coarse Ranking results
import pandas as pd
df_evaluate = pd.read_csv("./cross_encoder_eval.csv")

dev_samples = []
for index, row in df_evaluate.iterrows():
    query_text = row["query"]
    passage_text = row["passage"]
    label = int(row["is_relevent"])
    dev_samples.append(InputExample(texts = [query_text, passage_text], label = label))


print(len(dev_samples))


train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)

evaluator = CEBinaryClassificationEvaluator.from_input_examples(dev_samples, name='cross_encoder')

if args.do_train:
   # Train the model
   cross_encoder.fit(train_dataloader=train_dataloader,
          evaluator=evaluator,
          epochs=20,
          evaluation_steps=5000,
          warmup_steps=1000,
          save_best_model=True,
          output_path=model_save_path)


cross_encoder.evaluate(evaluator=evaluator)

'''
dev_q = list(dev_queries.items())[:10]
# dev_r = list(dev_rel_docs.items())[:10]

for (k,v) in dev_q:
  query = v
  passage_id = list(dev_rel_docs[k])[0]
  passage = dev_corpus[passage_id]
  print(query)
  print(passage_id)
  print(passage)
  query_embedding = model.encode(query)
  passage_embedding = model.encode(passage)

  print("Similarity:", util.pytorch_cos_sim(query_embedding, passage_embedding))

'''
