# coding=utf-8
# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from scipy.stats import mode

"""
This file contains the implementation of different aggregation functions to combine the outputs of the base
classifiers to give the final decision. 


References
----------
Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms. John Wiley & Sons, 2004.

Shipp, Catherine A., and Ludmila I. Kuncheva. "Relationships between combination methods and measures of diversity
in combining classifiers." Information fusion 3.2 (2002): 135-148.

Giacinto, Giorgio, and Fabio Roli. "Design of effective neural network ensembles for image classification purposes.
" Image and Vision Computing 19.9 (2001): 699-707.

"""


def majority_voting(classifier_ensemble, query):
    """Performs a majority voting combination scheme between the base classifiers
    specified in the vector indices. Returns the label of the query sample as the
    most voted class.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
                         containing the ensemble of classifiers used in the aggregation scheme.

    query : array of shape = [n_samples, n_features]
            containing the samples to be classified

    Returns
    -------
    predicted_label : array of shape = [n_samples]
                      The label of each query sample predicted using the majority voting rule
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
    classifier_ensemble : list of shape = [n_classifiers]
                         containing the ensemble of classifiers used in the aggregation scheme.

    weights : array of shape = [n_samples, n_classifiers]
              Weights associated to each base classifier for each sample


    query : array of shape = [n_samples, n_features]
            The samples to be classified

    Returns
    -------
    predicted_label : array of shape = [n_samples]
                      The label of each query sample predicted using the majority voting rule
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
    votes : array of shape = [n_samples, n_classifiers],
            The votes obtained by each classifier for each sample.

    Returns
    -------
    predicted_label : array of shape = [n_samples]
                      The label of each query sample predicted using the majority voting rule
    """
    return mode(votes, axis=1)[0]


def weighted_majority_voting_rule(votes, weights):
    """Applies the weighted majority voting rule based on the votes obtained.

    Parameters
    ----------
    votes : array of shape = [n_samples, n_classifiers],
            The votes obtained by each classifier for each sample.

    weights : array of shape = [n_samples, n_classifiers]
              Weights associated to each base classifier for each sample

    Returns
    -------
    predicted_label : array of shape = [n_samples]
                      The label of each query sample predicted using the majority voting rule
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

