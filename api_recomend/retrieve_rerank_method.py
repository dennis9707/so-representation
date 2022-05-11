import torch

import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import time
import gzip
import os
import torch


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--trained_bi_encoder", type=str)
parser.add_argument("--trained_cross_encoder", type=str)
parser.add_argument("--test_dataset", type=str)
args = parser.parse_args()


if not torch.cuda.is_available():
  print("Warning: No GPU found. Please add GPU to your notebook")


#We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
#model_name = 'msmarco-distilbert-base-v2'
model_name = './SBRT_output/training_biker_bi-encoder-min_5_max_10_ir_10_distilroberta-base-full-best/'
# model_name = "/content/drive/MyDrive/SBRT_output/training_biker_distilroberta_base_bi-encoder-min_50distilroberta-base-2021-02-23_02-33-56"
# model_name = "/content/drive/MyDrive/SBRT_output/training_biker_bi-encoder-min_5_max_10_ir_10_distilroberta-base_3_iter"
# model_name = "/content/drive/MyDrive/SBRT_output/training_biker_bi-encoder-min_5_max_10_ir_10_distilroberta-base-full-best"
bi_encoder = SentenceTransformer(args.trained_bi_encoder)
top_k = 100     #Number of passages we want to retrieve with the bi-encoder

#The bi-encoder will retrieve 100 documents. We use a cross-encoder, to re-rank the results list to improve the quality
#cross_encoder = CrossEncoder('SBRT_output/training_biker_cross-encoder-30_iter_TinyBERT-full-best/')
cross_encoder = CrossEncoder(args.trained_cross_encoder)

### Now we read the MS Marco dataset
data_folder = '../data/full_data_min_5_max_10_ir_10/'

os.makedirs(data_folder, exist_ok=True)
import json 

# in the order of 1 to 30k
corpus = []
collection_filepath = os.path.join(data_folder, 'Corpus_dict.json')
#"evaluate_Corpus_min_2_max_10.json"
with open(collection_filepath, 'r', encoding='utf8') as fIn:
  data = json.load(fIn)
  for k in range(len(data)):
    corpus.append(data[str(k)])

# in the order of 1 to 30k
Answers = []
collection_filepath = os.path.join(data_folder, 'Answers_dict.json')
with open(collection_filepath, 'r', encoding='utf8') as fIn:
  data = json.load(fIn)
  for k in range(len(data)):
    Answers.append(data[str(k)])


evaluate_corpus = []
evaluate_answers = []
collection_filepath = os.path.join(data_folder, 'evaluate_Corpus.json')
with open(collection_filepath, 'r', encoding='utf8') as fIn:
  data = json.load(fIn)
  for k in data:
    evaluate_corpus.append(corpus[k])
    evaluate_answers.append(Answers[k])



if args.test_dataset == 'random_test':
   ## random sampled 1k eval data --> Second Dataset
   queries = []
   queries_answers = []
   collection_filepath = os.path.join(data_folder, 'evaluate_queries.json')
   with open(collection_filepath, 'r', encoding='utf8') as fIn:
      data = json.load(fIn)
      for k in data:
         queries.append(corpus[k])
         queries_answers.append(Answers[k])
   print('Randomly eval queries', len(queries))

elif args.test_dataset == 'multi_api_test': 
  ## Test Set for Thrid Datasets "Multi-API" 
  queries = []
  queries_answers = []
  collection_filepath = os.path.join(data_folder, 'evaluate_multi_queries.json')
  with open(collection_filepath, 'r', encoding='utf8') as fIn:
     data = json.load(fIn)
     for k in data:
       queries.append(corpus[k])
       queries_answers.append(Answers[k])
  print('Randomly multi API queries', len(queries))

elif args.test_dataset == 'biker_test':
  import pandas as pd
  ## BIKER Maually Data
  df_test= pd.read_csv("../data/Biker_test_filtered.csv")
  # # df_test= pd.read_csv("/content/drive/MyDrive/biker_data/min_5_max_10_ir_10_30k/BIKER_querys_final.csv")
  queries = df_test["title"].to_list()
  queries_answers = df_test["answer"].to_list()
  queries_answers=[str(list(eval(x))) for x in queries_answers]
  print('BIKER test_queries', len(queries))


print(len(queries))
print(queries[:10])



filtered_evaluate_corpus =[]
filtered_evaluate_answers =[]
print(len(evaluate_corpus))
print(len(evaluate_answers))
for idx,q in enumerate(evaluate_corpus):
  if not q in queries:
    filtered_evaluate_corpus.append(evaluate_corpus[idx])
    filtered_evaluate_answers.append(evaluate_answers[idx])
evaluate_corpus = filtered_evaluate_corpus
evaluate_answers = filtered_evaluate_answers
print(len(evaluate_corpus))
print(len(evaluate_answers))
#   queries_answers[idx] = str(list(eval(queries_answers[idx])))
#   df=df[~(df["title"].isin(queries))]
# evaluate_corpus=df["title"].to_list()
# evaluate_answers=df["answer"].to_list()

passages = evaluate_corpus
# passages = corpus
corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)
# passages = evaluate_corpus
# passages = corpus

# We also compare the results to lexical search (keyword search). Here, we use 
# the BM25 algorithm which is implemented in the rank_bm25 package.

from rank_bm25 import BM25Okapi
#from sklearn.feature_extraction import stop_words
import string
from tqdm.autonotebook import tqdm
import numpy as np

# We lower case our text and remove stop-words from indexing
def bm25_tokenizer(text):
  tokenized_doc = []
  for token in text.lower().split():
    token = token.strip(string.punctuation)

    #if len(token) > 0 and token not in stop_words.ENGLISH_STOP_WORDS:
    if len(token) > 0: 
       tokenized_doc.append(token)
  return tokenized_doc

tokenized_corpus = []
for passage in tqdm(passages):
  tokenized_corpus.append(bm25_tokenizer(passage))

bm25 = BM25Okapi(tokenized_corpus)

#This function will search all wikipedia articles for passages that
#answer the query
def evaluate(query,answer):


  top_k=50
  answer=eval(answer)
  '''
  print(query, answer)
  '''
  bm25_scores = bm25.get_scores(bm25_tokenizer(query))
  top_n = np.argpartition(bm25_scores, -50)[-50:]
  bm25_hits = [{'corpus_id': idx, 'score': bm25_scores[idx]} for idx in top_n]
  bm25_hits = sorted(bm25_hits, key=lambda x: x['score'], reverse=True)

  BM25_counter = -1 
  BM25_tmp_map = 0
  BM25_tmp_mrr = 0
  temp_hits = 0
  tmep_answer = answer[:]
  for idx, hit in enumerate(bm25_hits[0:50]):
      candidate = eval(evaluate_answers[hit['corpus_id']].replace("\n", " "))
      for c in candidate:
        if c in tmep_answer:
          if BM25_counter == -1: BM25_counter = idx + 1
          temp_hits+=1
          
          BM25_tmp_map += temp_hits/(idx+1)
          tmep_answer.remove(c)

  # print(temp_hits)
  # print(BM25_tmp_map)
  BM25_tmp_map /= len(answer)
  BM25_tmp_mrr = 0.0
  if BM25_counter!= -1:
    BM25_tmp_mrr = 1/BM25_counter



  question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
  Encoder_hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
  Encoder_hits = Encoder_hits[0]  # Get the hits for the first query

  cross_inp = [[query, passages[hit['corpus_id']]] for hit in Encoder_hits]
  cross_scores = cross_encoder.predict(cross_inp)
  for idx in range(len(cross_scores)):
      Encoder_hits[idx]['cross-score'] = cross_scores[idx]

  
  Bi_Encoder_counter = -1 
  Bi_Encoder_tmp_map = 0
  Bi_Encoder_tmp_mrr = 0

  Bi_Encoder_hit_list=[0]*top_k
  Bi_Encoder_hit_recall_list=[0]*top_k

  temp_hits = 0
  Encoder_hits = sorted(Encoder_hits, key=lambda x: x['score'], reverse=True) 
  tmep_answer = answer[:]
  for idx,hit in enumerate(Encoder_hits[0:top_k]):
      candidate= eval(evaluate_answers[hit['corpus_id']].replace("\n", " "))
      for c in candidate:

        if c in answer:
          if not query==passages[hit['corpus_id']].replace("\n", " ").replace("?",""):
            Bi_Encoder_hit_list[idx]=1

        if c in tmep_answer:
          if not query==passages[hit['corpus_id']].replace("\n", " ").replace("?",""):
            

            if Bi_Encoder_counter == -1: 
              Bi_Encoder_counter = idx + 1

              '''
              print("\t{:.3f}\t{}\t{}".format(hit['score'], passages[hit['corpus_id']].replace("\n", " "), evaluate_answers[hit['corpus_id']].replace("\n", " ")))
              '''
            temp_hits+=1


            Bi_Encoder_hit_recall_list[idx]=1


            Bi_Encoder_tmp_map += temp_hits/(idx+1)
            tmep_answer.remove(c)
  Bi_Encoder_tmp_map /= len(answer)
  Bi_Encoder_tmp_mrr = 0.0
  if Bi_Encoder_counter!= -1:
    Bi_Encoder_tmp_mrr = 1/Bi_Encoder_counter

  # answer_api= Answers[hit['corpus_id']]
  Cross_Encoder_counter = -1 
  Cross_Encoder_tmp_map = 0
  Cross_Encoder_tmp_mrr = 0

  Cross_Encoder_hit_list=[0]*top_k
  Cross_Encoder_hit_recall_list=[0]*top_k

  temp_hits = 0
  Encoder_hits = sorted(Encoder_hits, key=lambda x: x['cross-score'], reverse=True)
  tmep_answer = answer[:]
  for idx,hit in enumerate(Encoder_hits[0:50]):
      # print("\t{:.3f}\t{}\t{}".format(hit['cross-score'], passages[hit['corpus_id']].replace("\n", " "), Answers[hit['corpus_id']].replace("\n", " ")))
      candidate= eval(evaluate_answers[hit['corpus_id']].replace("\n", " "))
      # print(candidate,answer)
      for c in candidate:

        if c in answer:
          if not query==passages[hit['corpus_id']].replace("\n", " ").replace("?",""):
            Cross_Encoder_hit_list[idx]=1

        if c in tmep_answer:

          if Cross_Encoder_counter == -1: Cross_Encoder_counter = idx + 1
          temp_hits+=1


          Cross_Encoder_hit_recall_list[idx]=1


          Cross_Encoder_tmp_map += temp_hits/(idx+1)
          tmep_answer.remove(c)
  Cross_Encoder_tmp_map /= len(answer)
  Cross_Encoder_tmp_mrr = 0.0
  if Cross_Encoder_counter!= -1:
    Cross_Encoder_tmp_mrr = 1/Cross_Encoder_counter



  return BM25_counter,BM25_tmp_mrr,BM25_tmp_map,Bi_Encoder_counter,Bi_Encoder_tmp_mrr,Bi_Encoder_tmp_map,Cross_Encoder_counter,Cross_Encoder_tmp_mrr,Cross_Encoder_tmp_map,str(answer),Bi_Encoder_hit_list,Bi_Encoder_hit_recall_list,Cross_Encoder_hit_list,Cross_Encoder_hit_recall_list

BM25_mrr = 0
BM25_map = 0

Bi_Encoder_mrr = 0
Bi_Encoder_map = 0

Cross_Encoder_mrr = 0
Cross_Encoder_map = 0

api_list =[]
good_result=[]


Bi_Encoder_precision=[0]*4
Bi_Encoder_recall=[0]*4

Cross_Encoder_precision=[0]*4
Cross_Encoder_recall=[0]*4

for idx in range(len(queries)):
# for idx in range(10):
  
  BM25_counter,BM25_tmp_mrr,BM25_tmp_map,Bi_Encoder_counter,Bi_Encoder_tmp_mrr,Bi_Encoder_tmp_map,Cross_Encoder_counter,Cross_Encoder_tmp_mrr,Cross_Encoder_tmp_map,answer_api,Bi_Encoder_hit_list,Bi_Encoder_hit_recall_list,Cross_Encoder_hit_list,Cross_Encoder_hit_recall_list = evaluate(query = queries[idx], answer = queries_answers[idx])
  '''
  print(BM25_counter, Bi_Encoder_counter, Cross_Encoder_counter)
  '''
  len_api = len(eval(answer_api))
  #print()
  
  temp_precision=[0]*4
  temp_recall=[0]*4
  for idx, n in enumerate([1,3,5,10]):
    temp_precision[idx] = sum(Bi_Encoder_hit_list[:n])/n
    temp_recall[idx] = sum(Bi_Encoder_hit_recall_list[:n])/(len_api)

  Bi_Encoder_precision = [x + y for (x, y) in zip(Bi_Encoder_precision, temp_precision)] 
  Bi_Encoder_recall = [x + y for (x, y) in zip(Bi_Encoder_recall, temp_recall)] 


  temp_precision=[0]*4
  temp_recall=[0]*4
  for idx, n in enumerate([1,3,5,10]):
    temp_precision[idx] = sum(Cross_Encoder_hit_list[:n])/n
    temp_recall[idx] = sum(Cross_Encoder_hit_recall_list[:n])/(len_api)

  Cross_Encoder_precision = [x + y for (x, y) in zip(Cross_Encoder_precision, temp_precision)] 
  Cross_Encoder_recall = [x + y for (x, y) in zip(Cross_Encoder_recall, temp_recall)] 


  # print(BM25_tmp_mrr, Bi_Encoder_tmp_mrr, Cross_Encoder_tmp_mrr)
  # print(BM25_tmp_map, Bi_Encoder_tmp_map, Cross_Encoder_tmp_map)
  # if -1<Bi_Encoder_counter < 3 or -1<Cross_Encoder_counter < 3:
  #   good_result.append([queries[idx], queries_answers[idx]])

  api_list.append(answer_api)
  BM25_mrr+=BM25_tmp_mrr
  BM25_map+=BM25_tmp_map

  Bi_Encoder_mrr+=Bi_Encoder_tmp_mrr
  Bi_Encoder_map+=Bi_Encoder_tmp_map
  
  Cross_Encoder_mrr+=Cross_Encoder_tmp_mrr
  Cross_Encoder_map+=Cross_Encoder_tmp_map

BM25_mrr/=len(queries)
BM25_map/=len(queries)

Bi_Encoder_mrr/=len(queries)
Bi_Encoder_map/=len(queries)

Cross_Encoder_mrr/=len(queries)
Cross_Encoder_map/=len(queries)

Bi_Encoder_precision = [x/len(queries) for x in Bi_Encoder_precision]
Bi_Encoder_recall = [x/len(queries) for x in Bi_Encoder_recall]


Cross_Encoder_precision = [x/len(queries) for x in Cross_Encoder_precision]
Cross_Encoder_recall = [x/len(queries) for x in Cross_Encoder_recall]

print("Bi_Encoder_precision @1 @3 @5 @10")
print(Bi_Encoder_precision)

print("Bi_Encoder_recall @1 @3 @5 @10")
print(Bi_Encoder_recall)

print("Cross_Encoder_precision @1 @3 @5 @10")
print(Cross_Encoder_precision)

print("Cross_Encoder_recall @1 @3 @5 @10")
print(Cross_Encoder_recall)

print('BM25_mrr,Bi_Encoder_mrr,Cross_Encoder_mrr')
print(BM25_mrr,Bi_Encoder_mrr,Cross_Encoder_mrr)
print('BM25_map,Bi_Encoder_map,Cross_Encoder_map')
print(BM25_map,Bi_Encoder_map,Cross_Encoder_map)
#print(len(list(set(api_list))))
#print(good_result)


