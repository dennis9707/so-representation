
## CLEAR Replication Package

The original replication of CLEAR is available [here](https://github.com/Moshiii/CLEAR-replication).
The original code of CLEAR is written in Jupyter Notebook. For simplicity, we turn it into python code files.


## Training Pre-trained Models on API Recommendation Task

CLEAR fine-tunes pre-trained models by using Sentence Transformer Library [here](https://www.sbert.net/docs/package_reference/SentenceTransformer.html). In the original paper, CLEAR uses distilled roberta-base to get the embedding of questions and APIs. To show the effectiveness of other pre-trained models in API recommendation task, we replace the distilled roberta-base by other pre-trained models such as Roberta-base, CodeBERT, and GraphCodeBERT. We keep the other parts the same and only replace the pre-trained models.

To train the models, please run:
```bash
python bi_encoder_train.py \
    --do_train \
    --model_name roberta-base \
    --batch_size 256 
```

## Testing Pre-trained Models on API Recommendation Task
To test the finetuned models, please run:
```bash
python retrieve_rerank_method.py \
    --trained_bi_encoder PATH_to_finetuned_model \
    --test_dataset biker_test
```



