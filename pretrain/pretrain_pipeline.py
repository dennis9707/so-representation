from transformers import LineByLineTextDataset
dataset = LineByLineTextDataset(
    tokenizer=tokenizer,
    file_path='./maMi/mami_roberta_pretrain.txt',
    block_size=512,
)

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=True, mlm_probability=0.15
)
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir="./roberta-retrained-mami",
    overwrite_output_dir=True,
    num_train_epochs=30,
    per_device_train_batch_size=4,
    save_steps=500,
    save_total_limit=2,
    seed=1,
)
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=dataset
)
trainer.train("./roberta-retrained-mami/checkpoint-23500")