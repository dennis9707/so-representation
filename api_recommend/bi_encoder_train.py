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


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--do_train", action="store_true")
parser.add_argument("--do_eval", action="store_true")
parser.add_argument("--eval_no_train", action="store_true")
parser.add_argument("--trained_model", type=str)
parser.add_argument("--batch_size", type=int)
args = parser.parse_args()



class TripletsDataset(IterableDataset):
    def __init__(self, model, queries, corpus, train_triplets):
        self.model = model
        self.queries = queries
        self.corpus = corpus
        self.train_triplets = train_triplets

    def __iter__(self):
        count = 0
        
        for triplet in self.train_triplets:
            qid, pos_id, neg_id = triplet
            qid = str(qid)
            pos_id = str(pos_id)
            neg_id = str(neg_id)
            query_text = self.corpus[qid]
            pos_text = self.corpus[pos_id]
            neg_text = self.corpus[neg_id]
            yield InputExample(texts=[query_text, pos_text, neg_text])

    def __len__(self):
        return len(train_triplets)


#### Just some code to print debug information to stdout
logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout
#train_batch_size = 64 #Increasing the train batch size improves the model performance, but requires more GPU memory
train_batch_size = args.batch_size

# The  model we want to fine-tune


model_name = 'distilroberta-base'
word_embedding_model = models.Transformer(model_name, max_seq_length=350)
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
model = SentenceTransformer(modules=[word_embedding_model, pooling_model])

# We construct the SentenceTransformer bi-encoder from scratch

# model = SentenceTransformer(model_name)

### Now we read the MS Marco dataset


for p in [15]:
  for n in [15]:
    p=str(p)
    n=str(n)

    #reset model
    model = SentenceTransformer(model_name)
    data_folder = '../data/full_data_min_5_max_10_ir_10/'
    os.makedirs(data_folder, exist_ok=True)
    import json 
    model_save_path = './SBRT_output/bi-encoder-min_5_max_'+p+'_ir_'+n+'_5k_distilroberta-base'+'-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
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




    eval_queries = {} 
    for k in evaluate_queries:
      eval_queries[str(k)] = corpus[str(k)]
    eval_Corpus = {} 
    for k in evaluate_Corpus:
      eval_Corpus[str(k)] = corpus[str(k)]

    dev_queries =  {str(k):v for k,v in eval_queries.items()}
    dev_corpus = {str(k):v for k,v in eval_Corpus.items()}
    dev_rel_docs = {str(k):set(str(v) for v in v[0]) for k,v in evaluate_rel_doc.items()}



    ir_evaluator = evaluation.InformationRetrievalEvaluator(dev_queries, corpus, dev_rel_docs, name='distilroberta-train_eval')

    # For training the SentenceTransformer model, we need a dataset, a dataloader, and a loss used for training.
    train_dataset = TripletsDataset(model=model, queries=corpus, corpus=corpus, train_triplets=train_triplets)
    train_dataloader = DataLoader(train_dataset, shuffle=False, batch_size=train_batch_size)
    train_loss = losses.MultipleNegativesRankingLoss(model=model)


    # Train the model
    model.fit(train_objectives=[(train_dataloader, train_loss)],
              evaluator=ir_evaluator,
              epochs=50,
              warmup_steps=1000,
              output_path=model_save_path,
              evaluation_steps=5000,
              save_best_model=True,
              use_amp=True)



if args.do_eval:
  model.evaluate(evaluator=ir_evaluator)


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
