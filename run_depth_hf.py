#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import numpy as np
import pandas as pd
import tensorflow as tf
from datasets import DatasetDict, Dataset, load_metric
from sklearn.metrics import classification_report
from transformers import *
from transformers import EarlyStoppingCallback

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["WANDB_DISABLED"] = "true"

print(len(tf.config.experimental.list_physical_devices('GPU')))
strategy = tf.distribute.OneDeviceStrategy(device="/gpu:1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", help="path to csv file containing dataset", required=True, type=str)
    parser.add_argument("--n_epochs", help="number of epochs", type=int, default=5)
    parser.add_argument("--pretrained_model", help="any registered huggingface model", type=str,
                        default='bert-base-cased')
    parser.add_argument("--context", help="name of column containing context (options: parent_text,thesis)", type=str,
                        default='None')
    parser.add_argument("--seed", type=str, help="random seed", default='34')
    parser.add_argument("--lr", type=str, help="learning rate", default='1e-5')
    parser.add_argument("--save_steps", help="number of update steps between two checkpoints", type=int, default=1000)
    parser.add_argument("--eval_steps", help="number of update steps between two evaluations", type=int, default=1000)
    parser.add_argument("--save_pred", help="save predictions flag", type=str, default='True')
    parser.add_argument("--output_dir", help="path to save model and predictions", required=True, type=str)
    parser.add_argument('--lr_scheduler_type', help='scheduler type to use', type=str, default='linear')
    parser.add_argument('--warmup_steps', help='number of steps for linear warmup', type=int,
                        default=10000)
    parser.add_argument('--weight_decay', help='strength of weight decay', type=float, default=0.1)
    parser.add_argument('--logging_dir', help='directory for storing training logs', type=str)
    parser.add_argument('--batch_size', help='size of batch', type=int, default=16)
    parser.add_argument('--max_seq_len', help='maximum length of sequence', type=int, default=128)
    args = parser.parse_args()

    LR = float(args.lr)
    SEED = int(args.seed)
    SAVE_PREDICTIONS = bool(args.save_pred)
    EPOCHS = args.n_epochs
    INPUT_DIR = args.input_data
    OUTPUT_DIR = args.output_dir
    MODEL = args.pretrained_model
    LOG_DIR = args.logging_dir
    MAX_SEQ_LEN = args.max_seq_len
    BATCH_SIZE = args.batch_size
    LR_SCHEDULER_TYPE = args.lr_scheduler_type
    WARMUP_STEPS = int(args.warmup_steps)
    WEIGHT_DECAY = float(args.weight_decay)
    SAVE_STEPS = int(args.save_steps)
    EVAL_STEPS = int(args.eval_steps)
    CONTEXT = None if args.context == 'None' else args.context

    metric = load_metric("glue", "mrpc")

    data = pd.read_csv(INPUT_DIR)
    data['revision_depth'] = data['max_revision_id'] - data['revision_id']
    data_4 = data[data.revision_depth == 4]
    CLAIM_IDS = data_4.claim_id.unique()

    data = data[data.claim_id.isin(CLAIM_IDS)]

    data.claim_text = data.claim_text.fillna('')
    data.parent_1_text = data.parent_1_text.fillna('')
    data.title = data.title.fillna('')

    data_1 = pd.concat([data[data.revision_depth == 1],
                        data[(data.claim_id.isin(data[data.revision_depth == 1].claim_id.unique())) & (
                                    data.revision_type == 'CLEAN')]
                        ]).reset_index(drop=True)
    data_2 = pd.concat([data[data.revision_depth == 2],
                        data[(data.claim_id.isin(data[data.revision_depth == 2].claim_id.unique())) & (
                                    data.revision_type == 'CLEAN')]
                        ]).reset_index(drop=True)

    data_3 = pd.concat([data[data.revision_depth == 3],
                        data[(data.claim_id.isin(data[data.revision_depth == 3].claim_id.unique())) & (
                                    data.revision_type == 'CLEAN')]
                        ]).reset_index(drop=True)
    data_4 = pd.concat([data[data.revision_depth == 1],
                        data[(data.claim_id.isin(data[data.revision_depth == 4].claim_id.unique())) & (
                                    data.revision_type == 'CLEAN')]
                        ]).reset_index(drop=True)

    datasets = [data_1, data_2, data_3, data_4]
    names = ['D1', 'D2', 'D3', 'D4']

    data_test_1 = data_1[data_1.data_split == 'test']
    data_test_2 = data_2[data_2.data_split == 'test']
    data_test_3 = data_3[data_3.data_split == 'test']
    data_test_4 = data_4[data_4.data_split == 'test']

    test_sets = [data_test_1, data_test_2, data_test_3, data_test_4]

    for rev_idx, temp_data in enumerate(datasets):

        data_dev = temp_data[(temp_data.data_split == 'dev')]
        data_train = temp_data[(temp_data.data_split == 'train')]

        print("Test: " + str(test_sets[rev_idx].shape[0]))
        print("Train: " + str(data_train.shape[0]))
        print("Dev: " + str(data_dev.shape[0]))

        tokenizer = AutoTokenizer.from_pretrained(MODEL)

        COLS = ['claim_text', 'label', 'claim_id', CONTEXT, 'rev_depth'] if CONTEXT else ['claim_text', 'label',
                                                                                          'claim_id', 'rev_depth']

        D = DatasetDict({"train": Dataset.from_pandas(data_train[COLS]),
                         "dev": Dataset.from_pandas(data_dev[COLS]),
                         "test_1": Dataset.from_pandas(data_test_1[COLS]),
                         "test_2": Dataset.from_pandas(data_test_2[COLS]),
                         "test_3": Dataset.from_pandas(data_test_3[COLS]),
                         "test_4": Dataset.from_pandas(data_test_4[COLS]),
                         })

        if CONTEXT:

            D = D.map(lambda e: tokenizer(e['claim_text'], e[CONTEXT], truncation=True, max_length=MAX_SEQ_LEN),
                      batched=True)
        else:

            D = D.map(lambda e: tokenizer(e['claim_text'], truncation=True, max_length=MAX_SEQ_LEN),
                      batched=True)

        D.set_format(type='tensorflow',
                     columns=['input_ids', 'token_type_ids', 'label', 'attention_mask'],
                     output_all_columns=False)

        num_training_steps = EPOCHS * len(D['train'])

        num_labels = 2
        model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=num_labels).to('cuda')
        optimizer = AdamW(model.parameters(), lr=LR)

        training_args = TrainingArguments(
            output_dir=OUTPUT_DIR,
            num_train_epochs=EPOCHS,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE,
            warmup_steps=WARMUP_STEPS,
            weight_decay=WEIGHT_DECAY,
            logging_dir=LOG_DIR,
            do_eval=True,
            do_train=True,
            seed=SEED,
            learning_rate=LR,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            evaluation_strategy='steps',
            eval_steps=EVAL_STEPS,
            save_total_limit=1,
            load_best_model_at_end=True,
            metric_for_best_model='accuracy',
            save_steps=SAVE_STEPS,

        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=D['train'].shuffle(seed=0),
            eval_dataset=D['dev'],
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)]
        )

        trainer.train()

        for temp_test_idx, temp_test in enumerate(['test_1', 'test_2', 'test_3', 'test_4']):

            pred = trainer.predict(D[temp_test])

            logits = pred.predictions
            labels = pred.label_ids
            preds = np.argmax(logits, axis=-1)

            if SAVE_PREDICTIONS:
                tt = D[temp_test].to_pandas()[COLS]
                tt['pred_logit'] = tf.nn.softmax(logits, name=None).numpy()[:, 1]
                tt['pred_label'] = labels
                tt['pred_dense'] = preds

                tt.to_csv(OUTPUT_DIR + "/" + "D" + str(rev_idx + 1) + "_D" + str(temp_test_idx + 1) + "_pred.csv")

            print(classification_report(labels, preds))
