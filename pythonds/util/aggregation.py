# coding=utf-8

import numpy as np
# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause
from scipy.stats import mode


def majority_voting(classifier_ensemble, query):
    """Performs a majority voting combination scheme between the base classifiers
    specified in the vector indices. Returns the label of the query sample as the
    most voted class.

    Parameters
    ----------
    classifier_ensemble : The ensemble of classifiers to be used in the aggregation scheme

    query : Sample to be classified

    Returns
    -------
    predicted_label : The label of the query sample, predicted by the majority voting rule
    """
    # Check if a single sample was passed down to the function. In this case the sample must be converted to a 2D array.
    if query.ndim == 1:
        query = np.atleast_2d(query)

    n_samples = query.shape[0]
    votes = np.zeros((n_samples, len(classifier_ensemble)))
    for clf_index, clf in enumerate(classifier_ensemble):
        votes[:, clf_index] = clf.predict(query)
    predicted_label = majority_voting_rule(votes)

    return predicted_label


def weighted_majority_voting(classifier_ensemble, weights, query):
    """Performs a majority voting combination scheme between the base classifiers
    specified in the vector indices. Returns the label of the query sample as the
    most voted class.

    Parameters
    ----------
    classifier_ensemble : index of the base classifier to be used in the combination scheme

    weights : the weights associated to each classifier for the combination scheme

    query : Sample to be classified

    Returns
    -------
    predicted_label : The label of the query sample, predicted by the majority voting rule
    """
    # Check if a single sample was passed down to the function. In this case the sample must be converted to a 2D array.
    if query.ndim == 1:
        query = np.atleast_2d(query)

    n_samples = query.shape[0]
    votes = np.zeros((n_samples, len(classifier_ensemble)))
    for clf_index, clf in enumerate(classifier_ensemble):
        votes[:, clf_index] = clf.predict(query)

    predicted_label = weighted_majority_voting_rule(votes, weights)
    return predicted_label


def majority_voting_rule(votes):
    """Applies the majority voting rule to the estimated votes

    Parameters
    ----------
    votes : The ensemble of classifiers to be used in the aggregation scheme

    Returns
    -------
    predicted_label : The label of the query sample, predicted by the majority voting rule
    """
    return mode(votes, axis=1)[0]


def weighted_majority_voting_rule(votes, weights):
    """Applies the majority voting rule.

    Parameters
    ----------
    votes : predictions of the base classifiers

    weights : The weights associated to each classifier in the combination scheme

    Returns
    -------
    predicted_label : The label of the query sample, predicted by the majority voting rule
    """
    if weights.shape != votes.shape:
        raise ValueError('The size of the arrays votes and weights should be the same. weights = {0} '
                         'while votes = {1}' .format(len(weights), len(votes)))

    n_samples = votes.shape[0]
    labels_set = np.unique(votes)
    w_votes = np.zeros((n_samples, len(labels_set)))
    for idx in range(n_samples):

        for label in labels_set:
            w_votes[idx, np.where(labels_set == label)] = sum(weights[idx, votes[idx] == label])

    predicted_label = labels_set[np.argmax(w_votes, axis=1)]
    return predicted_label

