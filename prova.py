slug = 'distilbert/distilbert-base-uncased'

import huggingface_hub as hfh
from datasets import load_dataset
hfh.login()

dsname = 'cornell-movie-review-data/rotten_tomatoes'

full_dataset = load_dataset(path=dsname, cache_dir='./mydatasets')
trainset = full_dataset['train']
validset = full_dataset['validation']

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(slug)

def data_process(sample):
    return tokenizer(sample['text'], max_length=512, padding='max_length')

def prepare_dataset(ds):
    ds = ds.map(data_process)
    ds = ds.rename_column('label', 'labels')
    ds.set_format('pt', columns=['input_ids', 'attention_mask', 'labels'], output_all_columns=True)
    return ds

trainset = prepare_dataset(full_dataset['train'])
validset = prepare_dataset(full_dataset['validation'])

from transformers import DistilBertForSequenceClassification
model = DistilBertForSequenceClassification.from_pretrained(slug, num_labels=2).cuda()

import numpy as np
from sklearn.metrics import accuracy_score
from transformers import DataCollatorWithPadding, TrainingArguments, Trainer
from transformers import TrainerCallback

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

class TrainEvalCallback(TrainerCallback):
    def on_epoch_end(self, args, state, control, **kwargs):
        trainer = kwargs["trainer"]

        train_metrics = trainer.evaluate(
            eval_dataset=trainer.train_dataset,
            metric_key_prefix="train"
        )

        test_metrics = trainer.evaluate(
            eval_dataset=trainer.eval_dataset,
            metric_key_prefix="test"
        )

        print(train_metrics)
        print(test_metrics)

dc = DataCollatorWithPadding(tokenizer)

train_args = TrainingArguments(
    output_dir='./output',
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    logging_strategy='epoch',
    logging_steps=.5,
    eval_strategy='epoch',
    learning_rate=5e-5
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=trainset,
    eval_dataset=validset,
    data_collator=dc,
    compute_metrics=compute_metrics
)
trainer.train()