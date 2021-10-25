#!/usr/bin/python3

import sys
from sklearn.datasets import make_blobs
import csv


def generate_dataset():
    """
    Source: https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
    :return     dataset:
    """
    dataset, _ = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    return dataset


def save_dataset(dataset):
    """
    Save dataset to CSV
    :param      dataset:
    """
    with open("clustering.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(dataset)


def main(argv):
    dataset = generate_dataset()
    save_dataset(dataset)


if __name__ == "__main__":
    main(sys.argv)
