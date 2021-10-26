#!/usr/bin/python3
"""
based on: https://towardsdatascience.com/machine-learning-algorithms-part-9-k-means-example-in-python-f2ad05ed5203
"""
import sys
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans


def load_dataset():
    """
    Load dataset with from csv
    :return     dataset:        pandas dataset
    """
    dataset = pd.read_csv("../data/clustering/clustering.csv", header=None)
    return dataset


def elbow_method(dataset):
    """
    Apply the elbow method to get optimal number of clusters.
    Plots the elbow graph and asks the user to input the visual optimal.
    :param      dataset:        pandas dataset
    :return     n_clusters:     integer with number of clusters
    """
    wcss = []
    for i in range(1, 11):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(dataset)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1, 11), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

    n_clusters = int(input("Optimal clusters in 'Elbow Method'-plot: "))

    return n_clusters


def main(argv):
    dataset = load_dataset()
    n_clusters = elbow_method(dataset)
    model = KMeans(n_clusters=n_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)  # Define model
    labels = model.fit_predict(dataset)  # predict the labels of clusters.
    u_labels = np.unique(labels)  # Getting unique labels

    # plotting the results:
    for i in u_labels:
        plt.scatter(dataset[labels == i][0], dataset[labels == i][1], label=i)
    plt.legend()
    plt.show()


if __name__ == "__main__":
    main(sys.argv)
