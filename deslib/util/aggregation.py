# coding=utf-8
# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
from scipy.stats.mstats import mode
from sklearn.utils.validation import check_array

from deslib.util.prob_functions import softmax

"""
This file contains the implementation of different aggregation functions to
combine the outputs of the base
classifiers to give the final decision.

References
----------
Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms.
John Wiley & Sons, 2004.

J. Kittler, M. Hatef, R. P. W. Duin, J. Matas, On combining classifiers, IEEE
Transactions on Pattern Analysis and Machine Intelligence 20 (1998) 226–239.
"""


def majority_voting(classifier_ensemble, X):
    """Apply the majority voting rule to predict the label of each sample in X.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
        Containing the ensemble of classifiers used in the
        aggregation scheme.

    X : array of shape (n_samples, n_features)
        The input data.

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    votes = _get_ensemble_votes(classifier_ensemble, X)
    predicted_label = majority_voting_rule(votes)

    return predicted_label


def weighted_majority_voting(classifier_ensemble, weights, X):
    """Apply the weighted majority voting rule to predict the label of each
    sample in X. The size of the weights vector should be equal to the size of
    the ensemble.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
        Containing the ensemble of classifiers used in the aggregation scheme.

    weights : array of shape (n_samples, n_classifiers)
              Weights associated to each base classifier for each sample


    X : array of shape (n_samples, n_features)
        The input data.

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    votes = _get_ensemble_votes(classifier_ensemble, X)
    predicted_label = weighted_majority_voting_rule(votes, weights)
    return predicted_label


def _get_ensemble_votes(classifier_ensemble, X):
    """Calculates the votes obtained by each based classifier in the ensemble
    for sample in X

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
        Containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape (n_samples, n_features)
        The input data.

    Returns
    -------
    votes : array of shape (n_samples, n_classifiers)
            The votes obtained by each base classifier
    """
    # Check if a single sample was passed down to the function. In this case
    # the sample must be converted to a 2D array.
    X = check_array(X, ensure_2d=False)
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
    votes : array of shape (n_samples, n_classifiers),
        The votes obtained by each classifier for each sample.

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    # Omitting nan value in the predictions as they comes from removed
    # classifiers
    return mode(votes, axis=1)[0][:, 0]


def weighted_majority_voting_rule(votes, weights, labels_set=None):
    """Applies the weighted majority voting rule based on the votes obtained by
    each base classifier and their
    respective weights.

    Parameters
    ----------
    votes : array of shape (n_samples, n_classifiers),
        The votes obtained by each classifier for each sample.

    weights : array of shape (n_samples, n_classifiers)
        Weights associated to each base classifier for each sample

    labels_set : (Default=None) set with the possible classes in the problem.

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    w_votes, labels_set = get_weighted_votes(votes, weights, labels_set)
    predicted_label = labels_set[np.argmax(w_votes, axis=1)]
    return predicted_label


def get_weighted_votes(votes, weights, labels_set=None):
    if weights.shape != votes.shape:
        raise ValueError(
            'The shape of the arrays votes and weights should be the '
            'same. weights = {} '
            'while votes = {}'.format(weights.shape, votes.shape))
    if labels_set is None:
        labels_set = np.unique(votes.astype(int))

    n_samples = votes.shape[0]
    w_votes = np.zeros((len(labels_set), n_samples))
    ma_weights = weights.view(np.ma.MaskedArray)

    for ind, label in enumerate(labels_set):
        ma_weights.mask = votes != label
        w_votes[ind, :] = ma_weights.sum(axis=1)

    return w_votes.T, labels_set


def sum_votes_per_class(predictions, n_classes):
    """Sum the number of votes for each class. Accepts masked arrays as input.

    Parameters
    ----------
    predictions : array of shape (n_samples, n_classifiers),
        The votes obtained by each classifier for each sample. Can be a masked
        array.

    n_classes : int
        Number of classes.

    Returns
    -------
    summed_votes : array of shape (n_samples, n_classes)
        Summation of votes for each class
    """
    votes = np.zeros((predictions.shape[0], n_classes), dtype=int)
    for label in range(n_classes):
        votes[:, label] = np.sum(predictions == label, axis=1)
    return votes


def _get_ensemble_probabilities(classifier_ensemble, X,
                                estimator_features=None):
    """Get the probabilities estimate for each base classifier in the ensemble

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
        Containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape (n_samples, n_features)
        The input data.

    estimator_features : array of shape (n_classifiers, n_selected_features)
        Indices containing the features used by each classifier.

    Returns
    -------
    list_proba : array of shape (n_samples, n_classifiers, n_classes)
        Probabilities predicted by each base classifier in the ensemble for all
        samples in X.
    """
    list_proba = []
    if estimator_features is None:
        for idx, clf in enumerate(classifier_ensemble):
            list_proba.append(clf.predict_proba(X))
    else:
        for idx, clf in enumerate(classifier_ensemble):
            list_proba.append(clf.predict_proba(X[:, estimator_features[idx]]))
    # transpose the array to have the
    # shape = [n_samples, n_classifiers, n_classes]
    return np.array(list_proba).transpose((1, 0, 2))


def predict_proba_ensemble(classifier_ensemble, X, estimator_features=None):
    """Estimates the posterior probabilities of the give ensemble for each
    sample in X.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
        Containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape (n_samples, n_features)
        The input data.

    estimator_features : array of shape (n_classifiers, n_selected_features)
        Indices containing the features used by each classifier.

    Returns
    -------
    predicted_proba : array of shape (n_samples, n_classes)
        Posterior probabilities estimates for each samples in X.
    """
    ensemble_proba = _get_ensemble_probabilities(classifier_ensemble,
                                                 X,
                                                 estimator_features)
    n_classifiers = ensemble_proba.shape[1]
    predicted_proba = np.sum(ensemble_proba, axis=1) / n_classifiers
    return predicted_proba


def aggregate_proba_ensemble_weighted(ensemble_proba, weights):
    predicted_proba = ensemble_proba * np.expand_dims(weights, axis=2)
    predicted_proba = predicted_proba.mean(axis=1)

    return softmax(predicted_proba)


def average_combiner(classifier_ensemble, X):
    """Ensemble combination using the Average rule.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
        Containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape (n_samples, n_features)
        The input data.

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    ensemble_proba = _get_ensemble_probabilities(classifier_ensemble, X)
    return average_rule(ensemble_proba)


def product_combiner(classifier_ensemble, X):
    """Ensemble combination using the Product rule.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
        Containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape (n_samples, n_features)
        The input data.

    Returns
    -------
    predicted_label : array of shape = [n_classifiers, n_samples, n_classes]
        Probabilities predicted by each base classifier in the ensemble for all
        samples in X.
    """
    ensemble_proba = _get_ensemble_probabilities(classifier_ensemble, X)
    return product_rule(ensemble_proba)


def maximum_combiner(classifier_ensemble, X):
    """Ensemble combination using the Maximum rule.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
        Containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape (n_samples, n_features)
        The input data.

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    ensemble_proba = _get_ensemble_probabilities(classifier_ensemble, X)
    return maximum_rule(ensemble_proba)


def minimum_combiner(classifier_ensemble, X):
    """Ensemble combination using the Minimum rule.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
        Containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape (n_samples, n_features)
        The input data.

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    ensemble_proba = _get_ensemble_probabilities(classifier_ensemble, X)
    return minimum_rule(ensemble_proba)


def median_combiner(classifier_ensemble, X):
    """Ensemble combination using the Median rule.

    Parameters
    ----------
    classifier_ensemble : list of shape = [n_classifiers]
        Containing the ensemble of classifiers used in the aggregation scheme.

    X : array of shape (n_samples, n_features)
        The input data.

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    ensemble_proba = _get_ensemble_probabilities(classifier_ensemble, X)
    return median_rule(ensemble_proba)


def average_rule(predictions):
    """Apply the average fusion rule to the predicted vector of class supports
    (predictions).

    Parameters
    ----------
    predictions : np array of shape (n_samples, n_classifiers, n_classes)
        Vector of class supports predicted by each base classifier for sample

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    _check_predictions(predictions)
    average_predictions = np.mean(predictions, axis=1)
    return np.argmax(average_predictions, axis=1)


def product_rule(predictions):
    """Apply the product fusion rule to the predicted vector of class supports
    (predictions).

    Parameters
    ----------
    predictions : array of shape (n_samples, n_classifiers, n_classes)
        Vector of class supports predicted by each base classifier for sample

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    _check_predictions(predictions)
    prod_predictions = np.prod(predictions, axis=1)
    return np.argmax(prod_predictions, axis=1)


def median_rule(predictions):
    """Apply the product fusion rule to the predicted vector of class supports
    (predictions).

    Parameters
    ----------
    predictions : np array of shape (n_samples, n_classifiers, n_classes)
        Vector of class supports predicted by each base classifier for sample

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    _check_predictions(predictions)
    median_predictions = np.median(predictions, axis=1)
    return np.argmax(median_predictions, axis=1)


def maximum_rule(predictions):
    """Apply the product fusion rule to the predicted vector of class supports
    (predictions).

    Parameters
    ----------
    predictions : np array of shape (n_samples, n_classifiers, n_classes)
        Vector of class supports predicted by each base classifier for sample

    Returns
    -------
    predicted_label : array of shape (n_samples)
        The label of each query sample predicted using the majority voting rule
    """
    _check_predictions(predictions)
    max_predictions = np.max(predictions, axis=1)
    return np.argmax(max_predictions, axis=1)


def minimum_rule(predictions):
    """Apply the product fusion rule to the predicted vector of class supports
    (predictions).

    Parameters
    ----------
    predictions : np array of shape (n_samples, n_classifiers, n_classes)
        Vector of class supports predicted by each base classifier for sample

    Returns
    -------
    list_proba : array of shape = [n_classifiers, n_samples, n_classes]
        Probabilities predicted by each base classifier in the ensemble for all
        samples in X.
    """
    _check_predictions(predictions)
    min_predictions = np.min(predictions, axis=1)
    return np.argmax(min_predictions, axis=1)


def _check_predictions(predictions):
    """Check if the predictions array has the correct size.

    Raises a value error if the array do not contain exactly 3 dimensions:
    [n_samples, n_classifiers, n_classes]

    """
    if predictions.ndim != 3:
        raise ValueError(
            'predictions must contain 3 dimensions: '
            '[n_samples, n_classifiers, n_classes]. Currently'
            'predictions has {} dimensions'.format(predictions.ndim))
