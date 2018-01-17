from unittest.mock import MagicMock

import numpy as np

# ----- Test Example #1 ------

# ex1: The distribution of samples of a test example.
X_dsel_ex1 = np.array([[-1, 1], [-0.75, 0.5], [-1.5, 1.5],
                       [1, 1], [0.75, 0.5], [1.5, 1.5],
                       [1, -1], [-0.5, 0.5], [0.5, 0.5],
                       [0, -1], [0.75, -0.5], [0.0, 0.0],
                       [-1, -1], [0, -0.5], [1, -1]])

# Labels associated with the samples. This information is used by techniques based on a posteriori information (LCA and
# a posteriori in the calculation of the competence level).
y_dsel_ex1 = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])

# Pre-processed results on DSEL. This information is used by the majority of DS techniques to estimate the competence.
dsel_processed_ex1 = np.array([[1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1], [0, 0, 0], [1, 1, 1],
                               [1, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 1, 1], [1, 1, 1]])

# pre-calculated indices of 7 Nearest neighbors used to estimate the competence level
neighbors_ex1 = np.array([[8, 11,  4,  7, 13, 10,  1],
                         [7,  1, 11, 13,  0,  8,  4],
                         [5,  3,  4,  8, 10, 11,  7]])

# Scores obtained for the two classes. This information is used by the techniques based on posterior probabilities
dsel_scores_ex1 = np.array([[1.0, 0.0, 0.5, 0.5, 0.8, 0.2],
                            [1.0, 0.0, 0.5, 0.5, 0.8, 0.2],
                            [1.0, 0.0, 0.5, 0.5, 0.8, 0.2],
                            [1.0, 0.0, 0.5, 0.5, 0.8, 0.2],
                            [1.0, 0.0, 0.5, 0.5, 0.8, 0.2],
                            [1.0, 0.0, 0.5, 0.5, 0.8, 0.2],
                            [1.0, 0.0, 0.5, 0.5, 0.8, 0.2],
                            [1.0, 0.0, 0.5, 0.5, 0.8, 0.2],
                            [1.0, 0.0, 0.5, 0.5, 0.8, 0.2],
                            [1.0, 0.0, 0.5, 0.5, 0.8, 0.2],
                            [1.0, 0.0, 0.5, 0.5, 0.8, 0.2],
                            [1.0, 0.0, 0.5, 0.5, 0.8, 0.2],
                            [1.0, 0.0, 0.5, 0.5, 0.8, 0.2],
                            [1.0, 0.0, 0.5, 0.5, 0.8, 0.2],
                            [1.0, 0.0, 0.5, 0.5, 0.8, 0.2]])

# Distance information is used by the probabilistic techniques (des.probabilistic) as well as the MLA, A Priori and
# A Posteriori methods. Three values are considered: all zeros, all ones and the real distances calculated on the toy
# example.
distances_ex1 = np.array([[0.35355339, 0.35355339, 0.55901699, 0.79056942, 0.79056942, 0.90138782, 1.03077641],
                          [0.3,  0.39051248,  0.53851648,  0.86023253,  0.94339811, 1.04403065,  1.28549601],
                          [0.70710678,  1.41421356,  1.95256242,  2.12132034,  2.79508497, 2.82842712,  2.91547595]])


# ----- Test Example all ones ------

dsel_processed_all_ones = np.ones((15, 3))

dsel_scores_all_ones = np.ones((15, 6))

distances_all_ones = np.ones((3, 7))

# ----- Test Example all zeros ------

dsel_processed_all_zeros = np.zeros((15, 3))

dsel_scores_all_zeros = np.zeros((15, 6))

distances_all_zeros = np.zeros((3, 7))


# ----- Test Example from Combining pattern classifiers  ------

# This example is used to test the results of the A priori, A posteriori and MLA techniques
distances_ex_kuncheva = np.linspace(1, 15, num=15)

# 10 neighbors used in the example
neighbors_ex_kuncheva = np.linspace(0, 14, num=15, dtype=int)

# target class of each example. independent means that it should be used by the a priori technique
y_dsel_ex_kuncheva_independent = np.array([2, 1, 2, 2, 3, 1, 2, 1, 3, 3, 2, 1, 2, 2, 1]) - 1

# dependent means that it should be used by the a priori technique
y_dsel_ex_kuncheva_dependent = np.array([1, 0, 1, 1, 2, 0, 1, 0, 0, 2, 1, 2, 1, 1, 0])


# Predictions of the base classifier ci used to estimate its competence level
classifier_pred_ex_kuncheva = np.array([2, 3, 2, 2, 1, 1, 2, 2, 3, 3, 1, 2, 2, 2, 1]) - 1

# whether or not the base classifier made the correct prediction for each sample in dsel
dsel_processed_kuncheva = np.transpose(np.array([[1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1]]))

# scores obtained by each class by the base classifier ci. In this example we consider that the posteriori is always 1
# fo the predicted class
dsel_scores_ex_kuncheva = np.array([[0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [1.0, 0.0, 0.0],
                                   [1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 0.0, 1.0],
                                   [0.0, 0.0, 1.0],
                                   [1.0, 0.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [0.0, 1.0, 0.0],
                                   [1.0, 0.0, 0.0]])

k_ex_kuncheva = 15

n_classes_ex_kuncheva = 3

# ----- Routines to generate a pool of classifiers using MagicMock  ------


def create_base_classifier(return_value, return_prob=None):
    classifier = MagicMock()
    classifier.predict.return_value = [return_value]
    classifier.predict_proba.return_value = return_prob
    return classifier


def create_pool_classifiers():
    clf_0 = create_base_classifier(return_value=0, return_prob=np.atleast_2d([0.5, 0.5]))
    clf_1 = create_base_classifier(return_value=1, return_prob=np.atleast_2d([1.0, 0.0]))
    clf_2 = create_base_classifier(return_value=0, return_prob=np.atleast_2d([0.33, 0.67]))
    pool_classifiers = [clf_0, clf_1, clf_2]
    return pool_classifiers


def create_pool_all_agree(return_value, size):
    return [create_base_classifier(return_value=return_value)] * size
