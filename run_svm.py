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
    parser.add_argument("--exp_setup", type=str, help="use predefined split (random) split or cross-category (cc) setting", default="random")
    args = parser.parse_args()

    SEED = int(args.seed)
    SAVE_PREDICTIONS = bool(args.save_pred)
    INPUT_DIR = args.input_data
    OUTPUT_DIR = args.output_dir
    MODEL = args.model
    EXP_SETUP = args.exp_setup
    CONTEXT = None if args.context == 'None' else args.context
    C_PARAM = float(args.C)
    MAX_ITER = int(args.max_iter)

    data = pd.read_csv(INPUT_DIR)

    claim_embeddings = initialize_emb(MODEL)

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    CATEGORIES = [
        'Children',
        'ClimateChange',
        'Democracy',
        'Economics',
        'Education',
        'Equality',
        'Ethics',
        'Europe',
        'Gender',
        'Government',
        'Health',
        'Justice',
        'Law',
        'Philosophy',
        'Politics',
        'Religion',
        'Science',
        'Society',
        'Technology',
        'USA'
    ]

    train_emb = get_embeddings(data[data.data_split == 'train'].claim_text.fillna('').values,
                               claim_embeddings,
                               'claim', MODEL)
    test_emb = get_embeddings(data[data.data_split == 'test'].claim_text.fillna('').values,
                              claim_embeddings,
                              'claim', MODEL)
    if CONTEXT:
        train_context_emb = get_embeddings(data[data.data_split == 'train'][CONTEXT].fillna('').values,
                                           claim_embeddings,
                                           'context')
        test_context_emb = get_embeddings(data[data.data_split == 'test'][CONTEXT].fillna('').values,
                                          claim_embeddings,
                                          'context')
        train_emb = pd.concat([train_emb, train_context_emb], axis=1)
        test_emb = pd.concat([test_emb, test_context_emb], axis=1)

    train_emb['label'] = data[data.data_split == 'train'].label.values
    test_emb['label'] = data[data.data_split == 'test'].label.values

    filter_cols = [col for col in train_emb if not col.startswith('label')]

    if EXP_SETUP == 'random':
        clf = LinearSVC(C=C_PARAM, random_state=SEED, max_iter=MAX_ITER)
        clf.fit(np.array(train_emb.sample(frac=1, random_state=SEED)[filter_cols]),
                np.array(train_emb.sample(frac=1, random_state=SEED)['label']))

        pred = clf.predict(np.array(test_emb[filter_cols]))
        print(classification_report(test_emb.label.values, pred))

        if SAVE_PREDICTIONS:
            test = data[data.data_split == 'test']
            test['pred'] = pred
            test.to_csv(OUTPUT_DIR + '/pred.csv')
    else:
        for cat in CATEGORIES:
            cat_train = data[data.data_split == 'train'].reset_index(drop=True)
            cat_test = data[data.data_split == 'test'].reset_index(drop=True)
            train_ids = cat_train[(cat_train[cat] == 0)].index
            test_ids = cat_test[cat_test[cat] == 1].index

            clf = LinearSVC(C=C_PARAM, random_state=SEED, max_iter=MAX_ITER)
            clf.fit(np.array(train_emb.loc[train_ids, :].sample(frac=1, random_state=SEED)[filter_cols]),
                    np.array(train_emb.loc[train_ids, :].sample(frac=1, random_state=SEED)['label']))

            pred = clf.predict(np.array(test_emb.loc[test_ids, :][filter_cols]))
            print(cat)
            print(classification_report(np.array(test_emb.loc[test_ids, :]['label']), pred))

            output = cat_test.loc[test_ids, :]
            output['pred'] = pred
            output.to_csv(OUTPUT_DIR + "/" + str(cat) + "_pred.csv")
