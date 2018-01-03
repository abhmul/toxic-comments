import argparse
import logging
import os
from functools import partial

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import nltk

from toxic_dataset import LABEL_NAMES, ToxicData
import process_data

parser = argparse.ArgumentParser(description='Run the NBSVM model.')
parser.add_argument('--seed', type=int, default=7, help="Seed fo the random number generator")
args = parser.parse_args()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

SEED = args.seed
np.random.seed(SEED)
TRAIN_ID = "nbsvm-3gram_%s" % SEED + "_kernel"

logging.info("Training model with id %s" % TRAIN_ID)


class NbSvmClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, C=1.0, dual='auto', verbose=0):
        self.C = C
        self.dual = dual
        self.verbose = verbose
        self._clf = None
        logging.info("Creating model with C=%s" % C)

    def predict(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict(x.multiply(self._r))

    def predict_proba(self, x):
        # Verify that model has been fit
        check_is_fitted(self, ['_r', '_clf'])
        return self._clf.predict_proba(x.multiply(self._r))

    def fit(self, x, y):
        # Check that X and y have correct shape
        x, y = check_X_y(x, y, accept_sparse=True)

        def pr(x, y_i, y):
            p = x[y == y_i].sum(0)
            return (p + 1) / ((y == y_i).sum() + 1)

        self._r = sparse.csr_matrix(np.log(pr(x, 1, y) / pr(x, 0, y)))
        x_nb = x.multiply(self._r)
        if self.dual == 'auto':
            self.dual = x_nb.shape[0] <= x_nb.shape[1]
        self._clf = LogisticRegression(C=self.C, dual=self.dual, n_jobs=1, verbose=self.verbose)
        self._clf.fit(x_nb, y)
        return self


(train_ids, train_texts, train_labels), (test_ids, test_texts) = process_data.load_data("../input")

# My version of cleaning the text
# clean_text = partial(process_data.clean_text, remove_stopwords=True, replace_numbers=True, remove_special_chars=True,
#                      stem_words=True)
# train_texts = process_data.clean(train_texts, clean_text)
# test_texts = process_data.clean(test_texts, clean_text)
# tokenize = nltk.tokenize.word_tokenize

# The kernel writer's version
import re, string
re_tok = re.compile('([{}“”¨«»®´·º½¾¿¡§£₤‘’])'.format(string.punctuation))


def tokenize(s):
    return re_tok.sub(r' \1 ', s).split()


logging.info("Training data length: %s" % len(train_texts))
logging.info("Test data length: %s" % len(test_texts))

# Split the data
logging.info("Splitting the data")
train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.1,
                                                                    random_state=np.random.randint(2**32))
logging.info("New training data length: %s" % len(train_texts))
logging.info("New validation data length: %s" % len(val_texts))
vec = TfidfVectorizer(ngram_range=(1, 3), tokenizer=tokenize,
                      min_df=3, max_df=0.9, strip_accents='unicode', use_idf=1,
                      smooth_idf=1, sublinear_tf=1)
logging.info("Created Vectorizer %s" % vec)
logging.info("Fitting to train docs...")
trn_term_doc = vec.fit_transform(train_texts)
logging.info("Transforming to val docs...")
val_term_doc = vec.transform(val_texts)
logging.info("Transforming to test docs...")
test_term_doc = vec.transform(test_texts)

# Search for the appropriate C
Cs = [1e-1, 5e-1, 7e-1, 8e-1, 9e-1, 1., 2., 3., 4., 7., 1e1]
best_models = [None for _ in LABEL_NAMES]
scores = [None for _ in LABEL_NAMES]
assert train_labels.shape[1] == len(LABEL_NAMES)
for i in range(train_labels.shape[1]):
    best_val = float("inf")
    best_C = None
    for C in Cs:
        logging.info("Fitting {} with C={}".format(LABEL_NAMES[i], C))
        model = NbSvmClassifier(C=C, verbose=0).fit(trn_term_doc, train_labels[:, i])
        # Evaluate the model
        val_preds = model.predict_proba(val_term_doc)[:, 1]
        score = log_loss(val_labels[:, i], val_preds)
        logging.info("Model had val score of %s" % score)
        if score < best_val:
            logging.info("New minimum score improved from {} to {}".format(best_val, score))
            best_models[i] = model
            best_val = score
            best_C = C
    scores[i] = best_val
    logging.info("Best score for {} with C={} is {}".format(LABEL_NAMES[i], best_C, scores[i]))
logging.info("Average Val Score is %s" % np.average(scores))

# Now with the best models, we run on the test data and produce our submission
test_preds = np.empty((test_ids.shape[0], len(LABEL_NAMES)))
logging.info("Creating test predictions...")
for i in range(train_labels.shape[1]):
    test_preds[:, i] = best_models[i].predict_proba(test_term_doc)[:, 1]


# Create the submission
logging.info("Saving the predictions in a submission file")
ToxicData.save_submission(os.path.join("../submissions", TRAIN_ID + ".csv"), test_ids, test_preds)