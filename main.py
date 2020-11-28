import argparse
import time

from core.utils.type_collection import EstimatorTypes
from core.train_test import train, test


def main(args):
    if args.mode == "train":
        # del args.mode
        train(**vars(args))
    elif args.mode == "test":
        del args.mode
        test(args)
    # if args.mode == 'train':
    #     train


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, default="samples/marketing/marketing_train.csv"
    )
    parser.add_argument("--target", type=str, default="insurance_subscribe")
    parser.add_argument("--problem_type", type=str, default="binary")
    parser.add_argument("--estimator", type=str, default=EstimatorTypes.DecisionTree)
    parser.add_argument("--mode", type=str, default="train")
    parser.add_argument("--save_path", type=str, default="results")
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--meta_path", type=str, default=None)

    args = parser.parse_args()
    main(args)
