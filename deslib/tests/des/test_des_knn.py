from unittest.mock import MagicMock

import numpy as np
import pytest

from deslib.des.des_knn import DESKNN
from deslib.tests.examples_test import create_pool_classifiers


def test_estimate_competence():
    """
    Test case:

    Correct labels: 0000111
    classifier 1:   1010000   (2/7 correct)
    classifier 2:   1000100   (4/7 correct)
    classifier 2:   0010110   (5/7 correct)

    Diversity: compute number of common errors (on both classifiers) and divide by 7:
    clf1 x clf2: 3/7
    clf1 x clf3: 2/7
    clf2 x clf3: 1/7

    clf1 diversity = (3+2)/7 = -5/7 (negative because we use the negative of double error)
    clf2 diversity = (3+1)/7 = -4/7
    clf3 diversity = (2+1)/7 = -3/7

    """

    x = np.array([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1])

    clf1 = create_base_classifier(np.array([1, 0, 1, 0, 0, 0, 0]))
    clf2 = create_base_classifier(np.array([1, 0, 0, 0, 1, 0, 0]))
    clf3 = create_base_classifier(np.array([0, 0, 1, 0, 1, 1, 0]))

    pool_classifiers = [clf1, clf2, clf3]

    target = DESKNN(pool_classifiers, k=7, pct_accuracy=1, pct_diversity=1)
    target.fit(x, y)
    target.DFP_mask = np.ones(target.n_classifiers)
    target._get_region_competence = lambda x: (None, [0, 1, 2, 3, 4, 5, 6])

    competences, diversity = target.estimate_competence(2)
    assert np.allclose(competences, [2./7, 4./7, 5./7])
    assert np.allclose(diversity, [-5./7, -4./7, -3./7])


def test_estimate_competence_Q():

    x = np.array([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1])

    clf1 = create_base_classifier(np.array([1, 0, 1, 0, 0, 0, 0]))
    clf2 = create_base_classifier(np.array([1, 0, 0, 0, 1, 0, 0]))
    clf3 = create_base_classifier(np.array([0, 0, 1, 0, 1, 1, 0]))

    pool_classifiers = [clf1, clf2, clf3]

    target = DESKNN(pool_classifiers, k=7, pct_accuracy=1, pct_diversity=1, metric='Q')
    target.fit(x, y)
    target.DFP_mask = np.ones(target.n_classifiers)
    target._get_region_competence = lambda x: (None, [0, 1, 2, 3, 4, 5, 6])

    competences, diversity = target.estimate_competence(2)
    assert np.allclose(competences, [2./7, 4./7, 5./7])
    assert np.allclose(diversity, [2, 1.2, 1.2])


def test_estimate_competence_ratio():

    x = np.array([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1])

    clf1 = create_base_classifier(np.array([1, 0, 1, 0, 0, 0, 0]))
    clf2 = create_base_classifier(np.array([1, 0, 0, 0, 1, 0, 0]))
    clf3 = create_base_classifier(np.array([0, 0, 1, 0, 1, 1, 0]))

    pool_classifiers = [clf1, clf2, clf3]

    target = DESKNN(pool_classifiers, k=7, pct_accuracy=1, pct_diversity=1, metric='Ratio')
    target.fit(x, y)
    target.DFP_mask = np.ones(target.n_classifiers)
    target._get_region_competence = lambda x: (None, [0, 1, 2, 3, 4, 5, 6])

    competences, diversity = target.estimate_competence(2)
    assert np.allclose(competences, [2./7, 4./7, 5./7])
    assert np.allclose(diversity, [2.166, 3.666, 4.500], atol=0.01)


def test_select():
    """
    Test case: 10 base classifiers; select 5 based on accuracy, then the 3 most diverse
    accuracies (/10): 4 6 1 2 9 8 7 9 3 2   (should select indices 1, 4, 5, 6, 7)
    diversity:        0 8 0 0 1 6 7 2 0 0   (should select indices 1, 5, 6 as most diverse)

    """
    pool_classifiers = [create_base_classifier(1) for _ in range(10)]

    accuracies = np.array([4, 6, 1, 2, 9, 8, 7, 9, 3, 2]) / 10.
    diversity = np.array([0, 8, 0, 0, 1, 6, 7, 2, 0, 0])
    target = DESKNN(pool_classifiers, k=7, pct_accuracy=5./10, pct_diversity=3./10)
    target.estimate_competence = lambda x: (accuracies, diversity)

    selected_indices = target.select(2)

    assert set(selected_indices) == {1, 5, 6}


def test_select_less_diverse():
    """
    Test case: 10 base classifiers; select 5 based on accuracy, then the 3 most diverse
    accuracies (/10): 4 6 1 2 9 8 7 9 3 2   (should select indices 1, 4, 5, 6, 7)
    diversity:        0 8 0 0 1 6 7 2 0 0   (should select indices 1, 5, 6 as most diverse)

    """
    pool_classifiers = [create_base_classifier(1) for _ in range(10)]

    accuracies = np.array([4, 6, 1, 2, 9, 8, 7, 9, 3, 2]) / 10.
    diversity = np.array([0, 8, 0, 0, 1, 6, 7, 2, 0, 0])
    target = DESKNN(pool_classifiers, k=7, pct_accuracy=5./10, pct_diversity=3./10, more_diverse=False)
    target.estimate_competence = lambda x: (accuracies, diversity)

    selected_indices = target.select(2)

    assert set(selected_indices) == {4, 5, 7}


def test_input_diversity_parameter():
    with pytest.raises(ValueError):
        DESKNN([create_base_classifier(1)]*100, metric='abc')


def test_J_N_values():
    with pytest.raises(ValueError):
        DESKNN([create_base_classifier(1)]*100, pct_accuracy=0.5, pct_diversity=0)


def test_J_higher_than_N():
    with pytest.raises(ValueError):
        DESKNN([create_base_classifier(1)]*100, pct_accuracy=0.3, pct_diversity=0.5)


def create_base_classifier(value):
    classifier = MagicMock()
    classifier.predict.return_value = value
    classifier.predict_proba.return_value = value

    return classifier


# Since the majority of the base classifiers selected predicts class 0, the final decision of the ensemble should be 0.
def test_classify_instance():
    query = np.atleast_2d([1, -1])
    des_knn_test = DESKNN(create_pool_classifiers() * 4, k=2)
    des_knn_test.select = MagicMock(return_value=[0, 1, 2, 3, 5, 6, 7, 9])

    predicted = des_knn_test.classify_instance(query)
    assert predicted == 0
