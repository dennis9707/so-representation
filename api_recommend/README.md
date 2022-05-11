
## CLEAR Replication Package

The original replication of CLEAR is available [here](https://github.com/Moshiii/CLEAR-replication).
The original code of CLEAR is written in Jupyter Notebook. For simplicity, we turn it into python code files.


## Training CLEAR 

The following example fine-tunes RoBERTa on WikiText-2. Here too, we're using the raw WikiText-2. The loss is different
as BERT/RoBERTa have a bidirectional mechanism; we're therefore using the same loss that was used during their
pre-training: masked language modeling.
In accordance to the RoBERTa paper, we use dynamic masking rather than static masking. The model may, therefore,
converge slightly slower (over-fitting takes more epochs).

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

