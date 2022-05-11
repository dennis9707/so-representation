
## CLEAR Replication Package

The original replication of CLEAR is available [here](https://github.com/Moshiii/CLEAR-replication).
The original code of CLEAR is written in Jupyter Notebook. For simplicity, we turn it into python code files.


## Training CLEAR 

The following example fine-tunes Pre-trained models by using Sentence Transformer Library [here](https://www.sbert.net/docs/package_reference/SentenceTransformer.html). CLEAR uses distilled roberta-base to get the embedding of questions and APIs. To show the effectiveness of other pre-trained models in API recommendation task, we replace the distilled roberta-base by other Pre-trained models such as Roberta-base, CodeBERT, GraphCodeBERT and keep the other parts the same.


```bash
python run_mlm.py \
    --model_name_or_path roberta-base \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm
```

