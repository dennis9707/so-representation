


import json
from sentence_transformers import SentenceTransformer, CrossEncoder, util
import time
import gzip
import os
import torch

if not torch.cuda.is_available():
  print("Warning: No GPU found. Please add GPU to your notebook")


#We use the Bi-Encoder to encode all passages, so that we can use it with sematic search
#model_name = 'msmarco-distilbert-base-v2'
model_name = 'SBRT_output/training_biker_bi-encoder-min_5_max_10_ir_10_distilroberta-base-full-best/'


# model_name = "/content/drive/MyDrive/SBRT_output/training_biker_distilroberta_base_bi-encoder-min_50distilroberta-base-2021-02-23_02-33-56"
# model_name = "/content/drive/MyDrive/SBRT_output/training_biker_bi-encoder-min_5_max_10_ir_10_distilroberta-base_3_iter"
# model_name = "/content/drive/MyDrive/SBRT_output/training_biker_bi-encoder-min_5_max_10_ir_10_distilroberta-base-full-best"
bi_encoder = SentenceTransformer(model_name)
top_k = 100     #Number of passages we want to retrieve with the bi-encoder


### Now we read the MS Marco dataset
p='10'
n='10'
data_folder = '../data/full_data_min_5_max_' +p+'_ir_'+n+'/'


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

'''
## random sampled 1k eval data
queries = []
queries_answers = []
collection_filepath = os.path.join(data_folder, 'evaluate_queries.json')
with open(collection_filepath, 'r', encoding='utf8') as fIn:
  data = json.load(fIn)
  for k in data:
    queries.append(corpus[k])
    queries_answers.append(Answers[k])
print('Randomly eval queries', len(queries))
'''

## BIKER Maually Data
import pandas as pd
df_test= pd.read_csv("../data/Biker_test_filtered.csv")
# # df_test= pd.read_csv("/content/drive/MyDrive/biker_data/min_5_max_10_ir_10_30k/BIKER_querys_final.csv")
queries = df_test["title"].to_list()
queries_answers = df_test["answer"].to_list()
queries_answers=[str(list(eval(x))) for x in queries_answers]
print('BIKER test_queries', len(queries))


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


## ----------- Embed all Eval Corpus

passages = evaluate_corpus
# passages = corpus
corpus_embeddings = bi_encoder.encode(passages, convert_to_tensor=True, show_progress_bar=True)
# passages = evaluate_corpus
# passages = corpus


## ----------- Coarse Ranking by Bi-encoder only

#This function will search all wikipedia articles for passages that
#answer the query
def get_cross_encoder_result(query,answer):

  ##### Sematic Search #####
  #Encode the query using the bi-encoder and find potentially relevant passages
  question_embedding = bi_encoder.encode(query, convert_to_tensor=True)
  # question_embedding = question_embedding.cuda()
  hits = util.semantic_search(question_embedding, corpus_embeddings, top_k=top_k)
  hits = hits[0]  # Get the hits for the first query

  #Output of top-5 hitt
  # print("Top-5 Bi-Encoder Retrieval hits")
  hits = sorted(hits, key=lambda x: x['score'], reverse=True)
  # print(len(hits))
  result_list = []
  added=False
  for idx,hit in enumerate(hits[0:50]):
      passage = passages[hit['corpus_id']]
      if evaluate_answers[hit['corpus_id']].replace("\n", " ") == answer:
        if added == False:
          result_list.append([query,passage,1])
        added=True
  counter=0
  for idx,hit in enumerate(hits[0:50]):
      passage = passages[hit['corpus_id']]
      if evaluate_answers[hit['corpus_id']].replace("\n", " ") != answer:
        if counter<10:
          result_list.append([query,passage,0])
        counter+=1

  return result_list
evaluate_dataset =[]
for idx in range(len(queries)):
  
  result_list = get_cross_encoder_result(query = queries[idx], answer = queries_answers[idx])
  evaluate_dataset+=result_list
df_cross_encoder = pd.DataFrame(data=evaluate_dataset,columns=["query","passage","is_relevent"])
df_cross_encoder.to_csv("./cross_encoder_eval.csv")
