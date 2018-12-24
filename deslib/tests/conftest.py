from unittest.mock import MagicMock
import numpy as np
import pytest


@pytest.fixture
def example_estimate_competence(create_X_y):
    X, y = create_X_y

    # Pre-processed results on DSEL. This information is used by the majority
    # of DS techniques to estimate the classifier competence.
    dsel_processed = np.array(
        [[1, 1, 1], [0, 0, 1], [0, 0, 1], [1, 1, 0], [1, 1, 1], [0, 0, 1],
         [0, 0, 0], [1, 1, 1],
         [1, 1, 0], [0, 0, 1], [0, 0, 1], [0, 1, 0], [0, 1, 0], [1, 1, 1],
         [1, 1, 1]])

    # pre-calculated indices of 7 Nearest neighbors for competence estimation.
    neighbors = np.array([[8, 11, 4, 7, 13, 10, 1],
                          [7, 1, 11, 13, 0, 8, 4],
                          [5, 3, 4, 8, 10, 11, 7]])

    # Scores obtained for the two classes. This information is used by the
    # techniques based on posterior probabilities
    dsel_scores = np.tile(np.array([[1.0, 0.0], [0.5, 0.5], [0.8, 0.2]]),
                          (15, 1, 1))

    # Distance information is used by the probabilistic techniques
    # (des.probabilistic) as well as the MLA, A Priori and A Posteriori methods
    distances = np.array([[0.35355339, 0.35355339, 0.55901699, 0.79056942,
                           0.79056942, 0.90138782, 1.03077641],
                          [0.3, 0.39051248, 0.53851648, 0.86023253, 0.94339811,
                           1.04403065, 1.28549601],
                          [0.70710678, 1.41421356, 1.95256242, 2.12132034,
                           2.79508497, 2.82842712, 2.91547595]])

    return X, y, neighbors, distances, dsel_processed, dsel_scores


@pytest.fixture
def create_X_y():
    # ex1: The distribution of samples of a test example.
    X = np.array([[-1, 1], [-0.75, 0.5], [-1.5, 1.5],
                  [1, 1], [0.75, 0.5], [1.5, 1.5],
                  [1, -1], [-0.5, 0.5], [0.5, 0.5],
                  [0, -1], [0.75, -0.5], [0.0, 0.0],
                  [-1, -1], [0, -0.5], [1, -1]])
    # Labels associated with the samples. This information is used
    # by techniques based on a posteriori information.
    y = np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    return X, y


# ----- Test Example all ones ------
@pytest.fixture
def example_all_ones(example_estimate_competence):
    X, y, neighbors = example_estimate_competence[0:3]
    dsel_processed = np.ones((15, 3))
    dsel_scores = np.ones((15, 3, 2))
    distances = np.ones((3, 7))

    return X, y, neighbors, distances, dsel_processed, dsel_scores


# ----- Test Example from Combining pattern classifiers  ------
# This example is used to test the results of the A priori,
# A posteriori and MLA techniques
@pytest.fixture
def example_kuncheva():
    distances = np.linspace(1, 15, num=15)

    # 10 neighbors used in the example
    neighbors = np.linspace(0, 14, num=15, dtype=int)

    # target class of each example. independent means that it should be
    # used by the a priori technique
    y_independent = np.array([2, 1, 2, 2, 3, 1, 2, 1, 3, 3, 2, 1, 2, 2, 1]) - 1

    # dependent means that it should be used by the a posteriori technique
    y_dependent = np.array([1, 0, 1, 1, 2, 0, 1, 0, 0, 2, 1, 2, 1, 1, 0])

    # Predictions of the base classifier ci. Used to estimate its competence
    # level for the A Posteriori
    classifier_pred = np.array(
        [2, 3, 2, 2, 1, 1, 2, 2, 3, 3, 1, 2, 2, 2, 1]) - 1

    # whether or not the base classifier made the correct prediction for each
    # sample in dsel
    dsel_processed = np.transpose(
        np.array([[1, 0, 1, 1, 0, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1]]))

    # In this example we consider that the posteriori is always 1 for the
    # predicted class (15 samples, 1 classifier, 3 classes)
    dsel_scores = np.array([[[0.0, 1.0, 0.0],
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
                             [1.0, 0.0, 0.0]]]).reshape(15, 1, 3)

    k = 15
    n_classes = 3
    dict_return = {"dsel_processed": dsel_processed,
                   "dsel_scores": dsel_scores,
                   "distances": distances,
                   "neighbors": neighbors,
                   "classifier_pred": classifier_pred,
                   "y_dependent": y_dependent,
                   "y_independent": y_independent,
                   "n_classes": n_classes,
                   "k": k}

    return dict_return


# ----- Routines to generate a pool of classifiers using MagicMock  ------
def create_base_classifier(return_value, return_prob=None):
    classifier = MagicMock()
    classifier.predict.return_value = [return_value]
    classifier.predict_proba.return_value = return_prob
    return classifier


@pytest.fixture
def create_pool_classifiers():
    clf_0 = create_base_classifier(return_value=0,
                                   return_prob=np.atleast_2d([0.5, 0.5]))
    clf_1 = create_base_classifier(return_value=1,
                                   return_prob=np.atleast_2d([1.0, 0.0]))
    clf_2 = create_base_classifier(return_value=0,
                                   return_prob=np.atleast_2d([0.33, 0.67]))
    pool_classifiers = [clf_0, clf_1, clf_2]
    return pool_classifiers


@pytest.fixture
def create_pool_all_agree():
    return [create_base_classifier(return_value=0)] * 100


@pytest.fixture
def example_static_selection(create_X_y):
    X, y = create_X_y
    pool1 = [create_base_classifier(return_value=0)] * 50
    pool2 = [create_base_classifier(return_value=1)] * 50
    for clf in pool1:
        clf.score = MagicMock(return_value=0.5)
    for clf in pool2:
        clf.score = MagicMock(return_value=1.0)

    pool = pool1 + pool2
    return X, y, pool
