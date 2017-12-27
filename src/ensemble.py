import argparse
import os
import pandas as pd

from toxic_dataset import ToxicData, LABEL_NAMES

parser = argparse.ArgumentParser(description='Train the models.')
parser.add_argument('-f', '--submission_fnames',required=True, type=str, nargs='+', help='The submission files to ensemble')
parser.add_argument('-w', '--weights', type=float, nargs='+', help='The weights for each submission file ' +
                                                                   '(default: uniform weighting).')
parser.add_argument('--custom_name', default=None, help="Allows using a custom name for the ensemble output")
args = parser.parse_args()


def ensemble_submissions(submission_fnames, weights=None):
    assert len(submission_fnames) > 0, "Must provide at least one submission to ensemble."

    if weights is None:
        weights = [1 / len(submission_fnames)] * len(submission_fnames)
    # Check that we have a weight for each submission
    assert len(submission_fnames) == len(weights), "Number of submissions and weights must match."
    # Normalize the weights
    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]
    # Get the id column of the submissions
    ids = pd.read_csv(submission_fnames[0])['id'].values
    # Read in all the submission values
    submissions = [pd.read_csv(sub_fname)[LABEL_NAMES].values for sub_fname in submission_fnames]
    # Combine them based on their respective weights
    combined = 0
    for weight, sub in zip(weights, submissions):
        combined = combined + weight * sub
    return ids, combined


def create_new_fname(submission_fnames, custom_name=None):
    fname_head = "../submissions/"
    if custom_name is None:
        fname_prefix = "ensemble"
        processed_names = [os.path.splitext(os.path.basename(subname))[0] for subname in submission_fnames]
        processed_names = set([param for name in processed_names for param in name[:name.rfind('_')].split('_')])
        ensemble_name = fname_prefix + "_".join(sorted(processed_names)) + ".csv"
    else:
        ensemble_name = custom_name
    return os.path.join(fname_head, ensemble_name)


if __name__ == "__main__":
    print(args.submission_fnames)
    ids, combined = ensemble_submissions(args.submission_fnames, weights=args.weights)
    ensemble_fname = create_new_fname(args.submission_fnames, custom_name=args.custom_name)
    ToxicData.save_submission(ensemble_fname, ids, combined)