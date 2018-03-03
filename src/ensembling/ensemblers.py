from copy import deepcopy

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.special import logit, expit
from xgboost import XGBClassifier

from pyjet.data import NpDataset

class Ensembler(BaseEstimator, ClassifierMixin):

    def fit(self, *args, **kwargs):
        raise NotImplementedError()

    def predict(self, *args, **kwargs):
        raise NotImplementedError()

    def score(self, *args, **kwargs):
        raise NotImplementedError()


class LogisticEnsembler(Ensembler):

    def __init__(self, penalty='l2', C=1.0, fit_intercept=False, verbose=0, n_jobs=-1):
        self.penalty = penalty
        self.C = C
        self.fit_intercept = fit_intercept
        self.verbose = verbose
        self.n_jobs = n_jobs
        self.model = LogisticRegression(penalty=penalty, C=C, fit_intercept=fit_intercept, verbose=verbose,
                                        n_jobs=n_jobs)

    def fit(self, X, y, *args, **kwargs):
        self.model.fit(X, y)

    def predict(self, X, *args, **kwargs):
        return self.model.predict_log_proba(X)[:, 1]

    def score(self, X, y, *args, **kwargs):
        return log_loss(y, self.model.predict_proba(X)[:, 1])


class XGBEnsembler(Ensembler):

    def __init__(self, max_depth=3, learning_rate=0.1,
                 n_estimators=100, silent=True, eval_metric='logloss',
                 early_stopping_rounds='auto',
                 objective="binary:logistic", booster='gbtree',
                 n_jobs=-1, nthread=None, gamma=0, min_child_weight=1,
                 max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1,
                 reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
                 base_score=0.5, random_state=0, seed=None, missing=None, **kwargs):
        self.model = XGBClassifier(max_depth, learning_rate, n_estimators, silent, objective, booster, n_jobs, nthread,
                                   gamma, min_child_weight, max_delta_step, subsample, colsample_bytree,
                                   colsample_bylevel, reg_alpha, reg_lambda, scale_pos_weight, base_score, random_state,
                                   seed, missing, **kwargs)
        self.eval_metric = eval_metric
        if early_stopping_rounds == 'auto':
            early_stopping_rounds = .1 * n_estimators
        self.early_stopping_rounds = early_stopping_rounds

    def fit(self, X, y, *args, validation=None, **kwargs):
        self.model.fit(expit(X), y, eval_set=validation,
                       early_stopping_rounds=self.early_stopping_rounds,
                       eval_metric=self.eval_metric)

    def predict(self, X, *args, **kwargs):
        # Take the expit of the X
        return logit(self.model.predict_proba(expit(X))[:, 1])

    def score(self, X, y, *args, **kwargs):
        return log_loss(y, self.model.predict_proba(expit(X))[:, 1])


class MultiLabelEnsembler(Ensembler):

    def __init__(self, *models):
        self.models = models
        self.num_labels = len(self.models)

    def fit(self, *args, **kwargs):
        raise ValueError("Fit is not defined. Models must be fitted first")

    def predict(self, X, *args, **kwargs):
        """
        :param X: n_samples X n_models x n_labels
        :param args: placeholder
        :param kwargs: placeholder
        :return: predictions as logits
        """
        assert self.num_labels == X.shape[2]
        return np.stack([self.models[i].predict(X[..., i], *args, **kwargs) for i in range(self.num_labels)], axis=-1)

    def score(self, X, y, *args, **kwargs):
        return np.average([model.score(X, y, *args, **kwargs) for model in self.models])


class KFoldEnsembler(Ensembler):

    def __init__(self, model, k):
        self.k = k
        self.models = [deepcopy(model) for _ in range(k)]
        self.__val_preds = None

    @property
    def val_preds(self):
        if self.__val_preds is None:
            raise ValueError("Must fit model before accessing val preds")
        return self.__val_preds

    def fit(self, X, y, *args, **kwargs):
        """
        :param X: n_samples X n_models
        :param y: n_samples
        :param args: placeholder
        :param kwargs: placeholder
        """
        data = NpDataset(X, y=y)
        self.__val_preds = np.zeros(X.shape)
        cur_sample_ind = 0
        for i, (train_data, val_data) in enumerate(data.kfold(self.k, shuffle=False)):
            self.models[i].fit(train_data.x, train_data.y, *args, validation=(val_data.x, val_data.y), **kwargs)
            # Predict and store the validation predictions
            val_preds = self.models[i].predict(val_data.x, *args, **kwargs)
            self.__val_preds[cur_sample_ind:cur_sample_ind + val_preds.shape[0]] = val_preds
            cur_sample_ind += val_preds.shape[0]

    def predict(self, X, *args, **kwargs):
        pred = 0.
        for model in self.models:
            pred = pred + model.predict(X, *args, **kwargs)
        return pred / self.k

    def score(self, X, y, *args, **kwargs):
        data = NpDataset(X, y=y)
        score = 0.
        for i, (train_data, val_data) in enumerate(data.kfold(self.k, shuffle=False)):
            score = score + self.models[i].score(val_data.x, val_data.y, *args, **kwargs)
        return score / self.k