# coding=utf-8
# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from scipy.stats import mode

from deslib.util.prob_functions import softmax

"""
This file contains the implementation of different aggregation functions to combine the outputs of the base
classifiers to give the final decision. 

References
----------
Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms. John Wiley & Sons, 2004.

J. Kittler, M. Hatef, R. P. W. Duin, J. Matas, On combining classifiers, IEEE Transactions on Pattern Analysis
and Machine Intelligence 20 (1998) 226–239.
"""


def majority_voting(classifier_ensemble, X):
    """Apply the majority voting rule to predict the label of each sample in X.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
                         containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape = [n_samples, n_features]
        The input data.

    Returns
    -------
    predicted_label : array of shape = [n_samples]
                      The label of each query sample predicted using the majority voting rule
    """
    votes = get_ensemble_votes(classifier_ensemble, X)
    predicted_label = majority_voting_rule(votes)

    return predicted_label


def weighted_majority_voting(classifier_ensemble, weights, X):
    """Apply the weighted majority voting rule to predict the label of each sample in X. The size of the weights
    vector should be equal to the size of the ensemble.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
                         containing the ensemble of classifiers used in the aggregation scheme.

    weights : array of shape = [n_samples, n_classifiers]
              Weights associated to each base classifier for each sample


    X : array of shape = [n_samples, n_features]
        The input data.

    Returns
    -------
    predicted_label : array of shape = [n_samples]
                      The label of each query sample predicted using the majority voting rule
    """
    votes = get_ensemble_votes(classifier_ensemble, X)
    predicted_label = weighted_majority_voting_rule(votes, weights)
    return predicted_label


def get_ensemble_votes(classifier_ensemble, X):
    """Calculates the votes obtained by each based classifier in the ensemble for sample in X

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
                         containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape = [n_samples, n_features]
        The input data.

    Returns
    -------
    votes : array of shape = [n_samples, n_classifiers]
            The votes obtained by each base classifier
    """
    # Check if a single sample was passed down to the function. In this case the sample must be converted to a 2D array.
    if X.ndim == 1:
        X = np.atleast_2d(X)

    n_samples = X.shape[0]
    votes = np.zeros((n_samples, len(classifier_ensemble)))
    for clf_index, clf in enumerate(classifier_ensemble):
        votes[:, clf_index] = clf.predict(X)

    return votes


def majority_voting_rule(votes):
    """Applies the majority voting rule to the estimated votes.

    Parameters
    ----------
    votes : array of shape = [n_samples, n_classifiers],
            The votes obtained by each classifier for each sample.

    Returns
    -------
    predicted_label : array of shape = [n_samples]
                      The label of each query sample predicted using the majority voting rule
    """
    return mode(votes, axis=1)[0][:, 0]


def weighted_majority_voting_rule(votes, weights):
    """Applies the weighted majority voting rule based on the votes obtained by each base classifier and their
    respective weights.

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
    if weights.ndim == 1:
        weights = np.atleast_2d(weights)

    if weights.size != votes.size:
        raise ValueError('The size of the arrays votes and weights should be the same. weights = {} '
                         'while votes = {}'.format(weights.size, votes.size))

    n_samples = votes.shape[0]
    labels_set = np.unique(votes)
    w_votes = np.zeros((n_samples, len(labels_set)))
    for idx in range(n_samples):

        for label in labels_set:
            w_votes[idx, np.where(labels_set == label)] = sum(weights[idx, votes[idx] == label])

    predicted_label = labels_set[np.argmax(w_votes, axis=1)]
    return predicted_label


def _get_ensemble_probabilities(classifier_ensemble, X):
    """Get the probabilities estimate for each base classifier in the ensemble

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
                         containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape = [n_samples, n_features]
        The input data.

    Returns
    -------
    list_proba : array of shape = [n_classifiers, n_samples, n_classes]
                 probabilities predicted by each base classifier in the ensemble for all samples in X.
    """
    list_proba = []
    for clf in classifier_ensemble:
        list_proba.append(clf.predict_proba(X))

    return np.array(list_proba)


def predict_proba_ensemble(classifier_ensemble, X):
    """Estimates the posterior probabilities of the give ensemble for each sample in X.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
                         containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape = [n_samples, n_features]
        The input data.

    Returns
    -------
    list_proba : array of shape = [n_classifiers, n_samples, n_classes]
                 probabilities predicted by each base classifier in the ensemble for all samples in X.
    """
    ensemble_proba = _get_ensemble_probabilities(classifier_ensemble, X)
    n_classifiers = ensemble_proba.shape[0]
    predicted_proba = np.einsum('ijk->jk', ensemble_proba) / n_classifiers
    return predicted_proba


def predict_proba_ensemble_weighted(classifier_ensemble, weights, X):
    """Estimates the posterior probabilities for each sample in X.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
                         containing the ensemble of classifiers used to estimate the probabilities.

    weights : array of shape = [n_samples, n_classifiers]
              Weights associated to each base classifier for each sample

    X : array of shape = [n_samples, n_features]
        The input data.

    Returns
    -------
    list_proba : array of shape = [n_classifiers, n_samples, n_classes]
                 probabilities predicted by each base classifier in the ensemble for all samples in X.
    """
    if weights.ndim == 1:
        weights = np.atleast_2d(weights)

    ensemble_proba = _get_ensemble_probabilities(classifier_ensemble, X)
    n_classifiers = ensemble_proba.shape[0]
    if n_classifiers != weights.shape[1]:
        raise ValueError('The number of weights should be equal to the number of base classifiers in the ensemble.'
                         'The number of classifiers is {},'
                         ' and the number of weights is {}' .format(n_classifiers, weights.shape[1]))

    predicted_proba = np.einsum('ijk,ji->jk', ensemble_proba, weights) / n_classifiers

    return softmax(predicted_proba)


def average_rule(classifier_ensemble, X):
    """Apply the average_rule rule to predict the label of each sample in X.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
                         containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape = [n_samples, n_features]
        The input data.

    Returns
    -------
    list_proba : array of shape = [n_classifiers, n_samples, n_classes]
                 probabilities predicted by each base classifier in the ensemble for all samples in X.
    """
    predicted_proba = predict_proba_ensemble(classifier_ensemble, X)
    return np.argmax(predicted_proba, axis=1)
