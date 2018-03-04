import argparse
import os
import logging
import pandas as pd
import numpy as np
import joblib
import copy

from sklearn.model_selection import ParameterGrid
from tqdm import trange
from joblib import Parallel

from pyjet.data import NpDataset
from toxic_dataset import ToxicData, LABEL_NAMES
from file_utils import safe_open_dir

from ensembling import ensemble_loading, ensemble_utils, ensemblers

parser = argparse.ArgumentParser(description='Train the models.')
parser.add_argument('ensemble_id', type=str, help='The id of the ensemble to create')
parser.add_argument('--seed', type=int, default=7, help="Seed fo the random number generator")
parser.add_argument('--kfold', type=int, default=10, help="Runs kfold validation with the input number")

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


class EnsembleTrainer(ensemblers.Ensembler):

    def __init__(self, ensemble_id, seed=7, k=10):
        self.ensemble_id = ensemble_id
        self.seed = seed
        self.k = k
        logging.info("Creating Ensemble {id} with seed {seed} and k {k}".format(id=self.ensemble_id, seed=self.seed,
                                                                                k=self.k))
        # Load the ensemble stuff
        self.constructor, self.submodel_names, self.params = ensemble_loading.get_ensemble_info(ensemble_id)
        self.grid = ParameterGrid(self.params)
        logging.info("Created a grid with parameters: {params}".format(params=self.params))
        self.base_dataset = ensemble_utils.create_predictions(self.submodel_names, self.k, self.seed)
        # Create the Ensembling model
        self.model = [[ensemblers.KFoldEnsembler(self.constructor(**param_set), self.k) for param_set in self.grid] for
                      _ in range(len(LABEL_NAMES))]
        self.best_model = ensemblers.MultiLabelEnsembler(
            *[self.model[label_num][0] for label_num in range(len(LABEL_NAMES))])

    def fit(self, *args, **kwargs):
        scores = np.zeros(len(LABEL_NAMES))
        rocs = np.zeros(len(LABEL_NAMES))
        accs = np.zeros(len(LABEL_NAMES))
        for label_num in range(len(LABEL_NAMES)):
            logging.info("Training for label {label}".format(label=LABEL_NAMES[label_num]))
            subdataset = NpDataset(self.base_dataset.x[..., label_num], y=self.base_dataset.y[..., label_num])
            # Best stuff
            best_score = float('inf')
            best_param_num = 0
            for param_num in trange(len(self.model[label_num])):
                # This will also save the val_preds
                self.model[label_num][param_num].fit(subdataset.x, subdataset.y, *args, **kwargs)

                # Save the model if its our best so far
                score = self.model[label_num][param_num].score(subdataset.x, subdataset.y)
                print(self.grid[param_num])
                if score < best_score:
                    logging.info("Score improved from {best_score} to {score}".format(best_score=best_score,
                                                                                      score=score))
                    self.best_model.models[label_num] = copy.deepcopy(self.model[label_num][param_num])
                    best_score = score
                    best_param_num = param_num
                    # Calculate the stats
                    scores[label_num] = best_score
                    rocs[label_num] = self.best_model.models[label_num].roc_auc(subdataset.x, subdataset.y)
                    accs[label_num] = self.best_model.models[label_num].accuracy(subdataset.x, subdataset.y)

                # Remove the current model from memory
                self.model[label_num][param_num] = None
            logging.info("Best score achieved is {best_score} with params {best_params}".format(best_score=best_score,
                                                                                                best_params=self.grid[best_param_num]))
            logging.info("Metrics are: Accuracy - {acc} --- ROC AUC - {roc}".format(acc=accs[label_num], roc=rocs[label_num]))

            logging.info("C")
        logging.info("Average Scores: LogLoss - {loss} --- Accuracy - {acc} --- ROC AUC - {roc}".format(loss=np.average(scores),
                                                                                                        acc=np.average(accs),
                                                                                                        roc=np.average(rocs)))

    def save_validation(self):
        val_preds = np.stack([model.val_preds for model in self.best_model.models], axis=-1)
        logging.info("Saving validation preds with shape {}".format(val_preds.shape))
        ensemble_utils.save_predictions(val_preds, self.base_dataset.y, self.ensemble_id, safe_open_dir("../superlearner_preds"))

    def predict(self, X, *args, **kwargs):
        return self.best_model.predict(X, *args, **kwargs)

    def save_model(self):
        save_path = os.path.join("../models", self.ensemble_id + ".dat")
        joblib.dump(self, save_path)
        logging.info("Saved model to " + save_path)

    @staticmethod
    def load_model(ensemble_id):
        load_path = os.path.join("../models", ensemble_id + ".dat")
        logging.info("Loading model from " + load_path)
        return joblib.load(load_path)

    def score(self, *args, **kwargs):
        raise NotImplementedError()

    def load_submissions(self):
        submission_fnames = [os.path.join("../submissions/", model_name + ".csv") for model_name in self.submodel_names]
        logging.info("Using submissions {submission_fnames}".format(submission_fnames=submission_fnames))
        # Get the ids
        ids = pd.read_csv(submission_fnames[0])['id'].values
        submissions = np.stack([pd.read_csv(sub_fname)[LABEL_NAMES].values for sub_fname in submission_fnames], axis=1)
        return ids, submissions

    def test(self, *args, **kwargs):
        ids, submissions = self.load_submissions()
        test_preds = self.predict(submissions, *args, **kwargs)
        return ids, test_preds


if __name__ == "__main__":
    args = parser.parse_args()
    parallel = Parallel(args.kfold, backend="threading", verbose=0)
    trainer = EnsembleTrainer(args.ensemble_id, args.seed, args.kfold)
    trainer.fit(parallel=parallel)
    trainer.save_model()
    trainer.save_validation()
    ids, test_preds = trainer.test()
    ToxicData.save_submission(os.path.join("../submissions", args.ensemble_id + ".csv"), ids, test_preds)
















