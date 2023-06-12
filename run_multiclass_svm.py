#!/usr/bin/env python
# coding: utf-8

import argparse
import os

from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

from utils.utils import *

if __name__ == "__main__":
    # Define arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_data", help="path to csv file containing dataset", required=True, type=str)
    parser.add_argument("--C", help="regularization parameter", type=float, default=0.1)
    parser.add_argument("--model",
                        help="model to use to generate embeddings(glove,flair, or registered huggingface transformer models) ", type=str,
                        default='glove')
    parser.add_argument("--context", help="name of column containing context (options: parent_text,thesis)", type=str,
                        default='None')
    parser.add_argument("--seed", type=str, help="random seed", default='34')
    parser.add_argument("--save_pred", help="save predictions flag", type=str, default='True')
    parser.add_argument("--max_iter", help="hard limit on iterations within solver", type=int, default=1000)
    parser.add_argument("--output_dir", help="path to save predictions", required=True, type=str)
    args = parser.parse_args()

    SEED = int(args.seed)
    SAVE_PREDICTIONS = bool(args.save_pred)
    INPUT_DIR = args.input_data
    OUTPUT_DIR = args.output_dir
    MODEL = args.model
    CONTEXT = None if args.context == 'None' else args.context
    C_PARAM = float(args.C)
    MAX_ITER = int(args.max_iter)

    data = pd.read_csv(INPUT_DIR)
    data['multiclass_label'] = data.revision_type.apply(get_multiclass_label)
    data = data[data.label.isin([0, 1, 2])].reset_index(drop=True)

    claim_embeddings = initialize_emb(MODEL)

    train_emb = get_embeddings(data[data.data_split == 'train'].claim_text.fillna('').values,
                               claim_embeddings,
                               'claim')
    test_emb = get_embeddings(data[data.data_split == 'test'].claim_text.fillna('').values,
                              claim_embeddings,
                              'claim')
    if CONTEXT:
        train_context_emb = get_embeddings(data[data.data_split == 'train'][CONTEXT].fillna('').values,
                                           claim_embeddings,
                                           'context')
        test_context_emb = get_embeddings(data[data.data_split == 'test'][CONTEXT].fillna('').values,
                                          claim_embeddings,
                                          'context')
        train_emb = pd.concat([train_emb, train_context_emb], axis=1)
        test_emb = pd.concat([test_emb, test_context_emb], axis=1)

    train_emb['multiclass_label'] = data[data.data_split == 'train'].multiclass_label.values
    test_emb['multiclass_label'] = data[data.data_split == 'test'].multiclass_label.values

    filter_cols = [col for col in train_emb if not col.startswith('multiclass_label')]

    clf = LinearSVC(C=C_PARAM, random_state=SEED, max_iter=MAX_ITER)
    clf.fit(np.array(train_emb.sample(frac=1, random_state=SEED)[filter_cols]),
            np.array(train_emb.sample(frac=1, random_state=SEED)['multiclass_label']))

    pred = clf.predict(np.array(test_emb[filter_cols]))
    print(classification_report(test_emb.multiclass_label.values, pred))

    if SAVE_PREDICTIONS:
        if not os.path.exists(OUTPUT_DIR):
            os.makedirs(OUTPUT_DIR)
        test = data[data.data_split == 'test']
        test['pred'] = pred
        test.to_csv(OUTPUT_DIR + '/pred.csv')
