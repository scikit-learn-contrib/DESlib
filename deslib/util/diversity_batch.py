import sys

import numpy as np

"""
This file contains the implementation of key diversity measures found in the
ensemble literature:

- Double Fault
- Negative Double fault
- Q-statistics
- Ratio of errors
- Agreement/Disagreement
- Classifier Correlation

The implementation are made according to the specifications from the book
"Combining Pattern Classifiers" based on Oracle outputs, i.e., taking into
account if the pair of classifiers made the correct/incorrect prediction:

N00 : represents samples that both classifiers made a wrong prediction

N10 : represents samples  that only classifier 2 predicts the wrong label.

N10 : represents samples  that only classifier 1 predicts the wrong label.

N11 :  represents samples  that both classifiers predicts the correct label.

References
----------
Kuncheva, Ludmila I. Combining pattern classifiers: methods and algorithms.
John Wiley & Sons, 2004.

Shipp, Catherine A., and Ludmila I. Kuncheva. "Relationships between
combination methods and measures of diversity in combining classifiers."
Information fusion 3.2 (2002): 135-148.

Giacinto, Giorgio, and Fabio Roli. "Design of effective neural network
ensembles for image classification purposes."
Image and Vision Computing 19.9 (2001): 699-707.

Aksela, Matti. "Comparison of classifier selection methods for improving
committee performance."
Multiple Classifier Systems (2003): 159-159.
"""


def _process_predictions(y: np.array, y_pred1: np.array,
                         y_pred2: np.array) -> np.array:
    """Pre-process the predictions of a pair of base classifiers for the
    computation of the diversity measures

    Parameters
    ----------
    y : array of shape (n_samples,):
        class labels of each sample.

    y_pred1 : array of shape (n_samples,):
              predicted class labels by the classifier 1 for each sample.


    y_pred2 : array of shape (n_classifiers, n_samples):
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    N00 : Array of shape (n_samples,)
        Percentage of samples that both classifiers predict the wrong label

    N10 : Array of shape (n_samples,)
        Percentage of samples that only classifier 2 predicts the wrong label

    N01 : Array of shape (n_samples,)
        Percentage of samples that only classifier 1 predicts the wrong label

    N11 : Array of shape (n_samples,)
        Percentage of samples that both classifiers predict the correct label
    """

    _, n_samples = y_pred2.shape

    if n_samples == 0:
        raise ValueError("Need at least one sample.")
    if n_samples != len(y_pred1) or n_samples != len(y):
        raise ValueError(
            "Inconsistent number of samples between "
            "class labels and predictions."
        )

    classifier_1_votes = y != y_pred1
    classifier_2_votes = y != y_pred2

    mismatch = classifier_1_votes != classifier_2_votes

    classifier_1_mismatch_error = np.logical_and(classifier_1_votes, mismatch)
    classifier_2_mismatch_error = np.logical_and(classifier_2_votes, mismatch)

    N01 = np.sum(classifier_1_mismatch_error, axis=1) / n_samples
    N10 = np.sum(classifier_2_mismatch_error, axis=1) / n_samples

    matching = np.invert(mismatch)
    n_matching = np.sum(matching, axis=1)

    both_incorrect = np.logical_and(classifier_1_votes, matching).sum(axis=1)

    N00 = both_incorrect / n_samples
    N11 = (n_matching - both_incorrect) / n_samples

    return N00, N10, N01, N11


def double_fault(y: np.array, y_pred1: np.array,
                 y_pred2: np.array) -> np.array:
    """Calculates the double fault (df) measure. This measure represents the
    probability that both classifiers makes the wrong prediction. A lower value
    of df means the base classifiers are less likely to make the same error.
    This measure must be minimized to increase diversity.

    Parameters
    ----------
    y : array of shape (n_samples,):
        class labels of each sample.

    y_pred1 : array of shape (n_samples,):
              predicted class labels by the classifier 1 for each sample.


    y_pred2 : array of shape (n_classifiers, n_samples):
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    df : The double fault measure between two classifiers

    References
    ----------
    Giacinto, Giorgio, and Fabio Roli. "Design of effective neural network
    ensembles for image classification purposes."
    Image and Vision Computing 19.9 (2001): 699-707.
    """
    N00, _, _, _ = _process_predictions(y, y_pred1, y_pred2)
    df = N00
    return df


def negative_double_fault(
        y: np.array, y_pred1: np.array, y_pred2: np.array
) -> np.array:
    """The negative of the double fault measure. This measure should be
    maximized for a higher diversity.

    Parameters
    ----------
    y : array of shape (n_samples,):
        class labels of each sample.

    y_pred1 : array of shape (n_samples,):
              predicted class labels by the classifier 1 for each sample.


    y_pred2 : array of shape (n_classifiers, n_samples):
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    df : The negative double fault measure between two classifiers

    References
    ----------
    Giacinto, Giorgio, and Fabio Roli. "Design of effective neural network
    ensembles for image classification purposes."
    Image and Vision Computing 19.9 (2001): 699-707.
    """
    return -double_fault(y, y_pred1, y_pred2)


def Q_statistic(y: np.array, y_pred1: np.array, y_pred2: np.array) -> np.array:
    """Calculates the Q-statistics diversity measure between a pair of
    classifiers. The Q value is in a range [-1, 1]. Classifiers that tend to
    classify the same object correctly will have positive values of Q, and
    Q = 0 for two independent classifiers.

    Parameters
    ----------
    y : array of shape (n_samples,):
        class labels of each sample.

    y_pred1 : array of shape (n_samples,):
              predicted class labels by the classifier 1 for each sample.


    y_pred2 : array of shape (n_classifiers, n_samples):
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    Q : The q-statistic measure between two classifiers
    """
    N00, N10, N01, N11 = _process_predictions(y, y_pred1, y_pred2)
    a = (N11 * N00) - (N01 * N10)
    b = (N11 * N00) + (N01 * N10)
    Q = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    return Q


def ratio_errors(y: np.array, y_pred1: np.array,
                 y_pred2: np.array) -> np.array:
    """Calculates Ratio of errors diversity measure between a pair of
    classifiers. A higher value means that the base classifiers are less likely
    to make the same errors. The ratio must be maximized for a higher diversity

    Parameters
    ----------
    y : array of shape (n_samples,):
        class labels of each sample.

    y_pred1 : array of shape (n_samples,):
              predicted class labels by the classifier 1 for each sample.


    y_pred2 : array of shape (n_classifiers, n_samples):
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    ratio : The q-statistic measure between two classifiers

    References
    ----------
    Aksela, Matti. "Comparison of classifier selection methods for improving
    committee performance."
    Multiple Classifier Systems (2003): 159-159.
    """
    N00, N10, N01, N11 = _process_predictions(y, y_pred1, y_pred2)
    N00[N00 == 0] = sys.float_info.max
    a = N01 + N10
    b = N00
    ratio = np.divide(a, b, out=np.zeros_like(a), where=b != 0)
    return ratio


def disagreement_measure(y: np.array, y_pred1: np.array,
                         y_pred2: np.array) -> np.array:
    """Calculates the disagreement measure between a pair of classifiers. This
    measure is calculated by the frequency that only one classifier makes the
    correct prediction.

    Parameters
    ----------
    y : array of shape (n_samples,):
        class labels of each sample.

    y_pred1 : array of shape (n_samples,):
              predicted class labels by the classifier 1 for each sample.


    y_pred2 : array of shape (n_classifiers, n_samples):
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    disagreement : The frequency at which both classifiers disagrees
    """
    _, N10, N01, _ = _process_predictions(y, y_pred1, y_pred2)
    disagreement = N10 + N01
    return disagreement


def agreement_measure(y: np.array, y_pred1: np.array,
                      y_pred2: np.array) -> np.array:
    """Calculates the agreement measure between a pair of classifiers. This
    measure is calculated by the frequency that both classifiers either
    obtained the correct or incorrect prediction for any given sample

    Parameters
    ----------
    y : array of shape (n_samples,):
        class labels of each sample.

    y_pred1 : array of shape (n_samples,):
              predicted class labels by the classifier 1 for each sample.


    y_pred2 : array of shape (n_classifiers, n_samples):
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    agreement : The frequency at which both classifiers agrees
    """
    N00, _, _, N11 = _process_predictions(y, y_pred1, y_pred2)
    agreement = N00 + N11
    return agreement


def correlation_coefficient(
        y: np.array, y_pred1: np.array, y_pred2: np.array
) -> np.array:
    """Calculates the correlation  between two classifiers using oracle
    outputs. Coefficient is a value in a range [-1, 1].

    Parameters
    ----------
    y : array of shape (n_samples,):
        class labels of each sample.

    y_pred1 : array of shape (n_samples,):
              predicted class labels by the classifier 1 for each sample.


    y_pred2 : array of shape (n_classifiers, n_samples):
              predicted class labels by the classifier 2 for each sample.

    Returns
    -------
    rho : The correlation coefficient measured between two classifiers
    """
    N00, N10, N01, N11 = _process_predictions(y, y_pred1, y_pred2)
    tmp = (N11 * N00) - (N10 * N01)
    b = np.sqrt((N11 + N01) * (N10 + N00) * (N11 + N10) * (N01 + N00))
    rho = np.divide(tmp, b, out=np.zeros_like(tmp), where=b != 0)
    return rho


def compute_pairwise_diversity(
        targets: np.array, prediction_matrix: np.array,
        diversity_func: np.array
) -> np.array:
    """Computes the pairwise diversity matrix.

    Parameters
    ----------
    targets : array of shape (n_samples):
       Class labels of each sample in X.

    prediction_matrix : array of shape (n_samples, n_classifiers):
       Predicted class labels for each classifier in the pool

    diversity_func : Function
       Function used to estimate the pairwise diversity

    Returns
    -------
    diversity : array of shape = [n_classifiers]
       The average pairwise diversity matrix calculated for the pool of
       classifiers

    """
    n_classifiers = prediction_matrix.shape[1]
    diversity = np.zeros(n_classifiers)

    for clf_index in range(n_classifiers):
        for clf_index2 in range(clf_index + 1, n_classifiers):
            this_diversity = diversity_func(
                targets,
                prediction_matrix[:, clf_index],
                prediction_matrix[:, clf_index2],
            )

            diversity[clf_index] += this_diversity
            diversity[clf_index2] += this_diversity

    return diversity
