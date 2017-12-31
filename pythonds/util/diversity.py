# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import sys

"""
This file contains the implementation of key diversity measures found in the ensemble literature:

- Double Fault 
- Negative Double fault
- Q-statistics
- Ratio of errors

The implementation are made according to the specifications from the book "Combining Pattern Classifiers".


References
----------
Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms. John Wiley & Sons, 2004.

Shipp, Catherine A., and Ludmila I. Kuncheva. "Relationships between combination methods and measures of diversity
in combining classifiers." Information fusion 3.2 (2002): 135-148.

Giacinto, Giorgio, and Fabio Roli. "Design of effective neural network ensembles for image classification purposes.
" Image and Vision Computing 19.9 (2001): 699-707.

Aksela, Matti. "Comparison of classifier selection methods for improving committee performance.
" Multiple Classifier Systems (2003): 159-159.
"""


def _process_predictions(y, y_pred1, y_pred2):
    """Pre-process the predictions of a pair of base classifiers for the computation of the diversity measures

    Parameters
    ----------
    y : array of shape = [n_samples]:
        class labels of each sample in X.

    y_pred1 : array of shape = [n_samples]:
              predicted class labels by the classifier 1 for each sample in X.

    y_pred2 : array of shape = [n_samples]:
              predicted class labels by the classifier 2 for each sample in X.

    Returns
    -------
    N00 : Percentage of samples that both classifiers predict the wrong label.

    N10 : Percentage of samples that only classifier 2 predicts the wrong label.

    N10 : Percentage of samples that only classifier 1 predicts the wrong label.

    N11 : Percentage of samples that both classifiers predicts the correct label.
    """
    size_y = len(y)
    if size_y != len(y_pred1) or size_y != len(y_pred2):
        raise ValueError('The vector with class labels must have the same size.')

    N00, N10, N01, N11 = 0.0, 0.0, 0.0, 0.0
    for index in range(size_y):
        if y_pred1[index] == y[index] and y_pred2[index] == y[index]:
            N11 += 1.0
        elif y_pred1[index] == y[index] and y_pred2[index] != y[index]:
            N10 += 1.0
        elif y_pred1[index] != y[index] and y_pred2[index] == y[index]:
            N01 += 1.0
        else:
            N00 += 1.0

    return N00/size_y, N10/size_y, N01/size_y, N11/size_y


def double_fault(y, y_pred1, y_pred2):
    """Calculates the double fault (df) measure. This measure represents the probability that both classifiers makes the
    wrong prediction. A lower value of df means the base classifiers are less likely to make the same error.
    This measure must be minimized to increase diversity.

    Parameters
    ----------
    y : array of shape = [n_samples]:
        class labels of each sample in X.

    y_pred1 : array of shape = [n_samples]:
              predicted class labels by the classifier 1 for each sample in X.

    y_pred2 : array of shape = [n_samples]:
              predicted class labels by the classifier 2 for each sample in X.

    Returns
    -------
    df : The double fault measure between two classifiers

    References
    ----------
    Giacinto, Giorgio, and Fabio Roli. "Design of effective neural network ensembles for image classification purposes."
    Image and Vision Computing 19.9 (2001): 699-707.
    """
    N00, _, _, _ = _process_predictions(y, y_pred1, y_pred2)
    df = N00
    return df


def negative_double_fault(y, y_pred1, y_pred2):
    """The negative of the double fault measure. This measure should be maximized for a higher diversity.

    Parameters
    ----------
    y : array of shape = [n_samples]:
        class labels of each sample in X.

    y_pred1 : array of shape = [n_samples]:
              predicted class labels by the classifier 1 for each sample in X.

    y_pred2 : array of shape = [n_samples]:
              predicted class labels by the classifier 2 for each sample in X.

    Returns
    -------
    df : The negative double fault measure between two classifiers

    References
    ----------
    Giacinto, Giorgio, and Fabio Roli. "Design of effective neural network ensembles for image classification purposes."
    Image and Vision Computing 19.9 (2001): 699-707.
    """
    return -double_fault(y, y_pred1, y_pred2)


def Q_statistic(y, y_pred1, y_pred2):
    """Calculates the Q-statistics diversity measure between a pair of classifiers. The Q value is in a range [-1, 1].
    Classifiers that tend to classify the same object correctly will have positive values of Q, and
    Q = 0 for two independent classifiers.

    Parameters
    ----------
    y : array of shape = [n_samples]:
        class labels of each sample in X.

    y_pred1 : array of shape = [n_samples]:
              predicted class labels by the classifier 1 for each sample in X.

    y_pred2 : array of shape = [n_samples]:
              predicted class labels by the classifier 2 for each sample in X.

    Returns
    -------
    Q : The q-statistic measure between two classifiers
    """
    N00, N10, N01, N11 = _process_predictions(y, y_pred1, y_pred2)
    Q = ((N11*N00) - (N01*N10)) / ((N11 * N00) + (N01 * N10))
    return Q


def ratio_errors(y, y_pred1, y_pred2):
    """Calculates Ratio of errors diversity measure between a pair of classifiers. A higher value means that the base
    classifiers are less likely to make the same errors. The ratio must be maximized for a higher diversity.

    Parameters
    ----------
    y : array of shape = [n_samples]:
        class labels of each sample in X.

    y_pred1 : array of shape = [n_samples]:
              predicted class labels by the classifier 1 for each sample in X.

    y_pred2 : array of shape = [n_samples]:
              predicted class labels by the classifier 2 for each sample in X.

    Returns
    -------
    ratio : The q-statistic measure between two classifiers

    References
    ----------
    Aksela, Matti. "Comparison of classifier selection methods for improving committee performance."
    Multiple Classifier Systems (2003): 159-159.
    """
    N00, N10, N01, N11 = _process_predictions(y, y_pred1, y_pred2)
    if N00 == 0:
        ratio = sys.float_info.max
    else:
        ratio = (N01 + N10) / N00
    return ratio

