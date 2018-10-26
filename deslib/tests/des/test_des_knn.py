from unittest.mock import MagicMock

import numpy as np
import pytest
from sklearn.linear_model import Perceptron

from deslib.des.des_knn import DESKNN
from sklearn.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(DESKNN)


def test_estimate_competence():
    """
    Test case:

    Correct labels: 0000111
    classifier 1:   1010000   (2/7 correct)
    classifier 2:   1000100   (4/7 correct)
    classifier 2:   0010110   (5/7 correct)

    Diversity: compute number of common errors (on both classifiers) and
    divide by 7:
    clf1 x clf2: 3/7
    clf1 x clf3: 2/7
    clf2 x clf3: 1/7

    clf1 diversity = (3+2)/7 = -5/7 (negative because we use the negative of
                                     double error)
    clf2 diversity = (3+1)/7 = -4/7
    clf3 diversity = (2+1)/7 = -3/7

    """
    query = np.ones((1, 2))
    x = np.array([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1])

    clf1 = create_base_classifier(np.array([1, 0, 1, 0, 0, 0, 0]))
    clf2 = create_base_classifier(np.array([1, 0, 0, 0, 1, 0, 0]))
    clf3 = create_base_classifier(np.array([0, 0, 1, 0, 1, 1, 0]))

    pool_classifiers = [clf1, clf2, clf3]

    target = DESKNN(pool_classifiers, k=7, pct_accuracy=1, pct_diversity=1)
    target.fit(x, y)
    neighbors = np.array([[0, 1, 2, 3, 4, 5, 6]])
    competences, diversity = target.estimate_competence(query, neighbors)
    assert np.allclose(competences, [2. / 7, 4. / 7, 5. / 7])
    assert np.allclose(diversity, [-5. / 7, -4. / 7, -3. / 7])


def test_estimate_competence_batch():
    """
    Test case:

    Correct labels: 0000111
    classifier 1:   1010000   (2/7 correct)
    classifier 2:   1000100   (4/7 correct)
    classifier 2:   0010110   (5/7 correct)

    Diversity: compute number of common errors (on both classifiers) and
    divide by 7:
    clf1 x clf2: 3/7
    clf1 x clf3: 2/7
    clf2 x clf3: 1/7

    clf1 diversity = (3+2)/7 = -5/7 (negative because we use the negative of
                                     double error)
    clf2 diversity = (3+1)/7 = -4/7
    clf3 diversity = (2+1)/7 = -3/7

    """

    n_samples = 10
    query = np.ones((n_samples, 2))
    x = np.array([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1])

    clf1 = create_base_classifier(np.array([1, 0, 1, 0, 0, 0, 0]))
    clf2 = create_base_classifier(np.array([1, 0, 0, 0, 1, 0, 0]))
    clf3 = create_base_classifier(np.array([0, 0, 1, 0, 1, 1, 0]))

    pool_classifiers = [clf1, clf2, clf3]

    target = DESKNN(pool_classifiers, k=7, pct_accuracy=1, pct_diversity=1)
    target.fit(x, y)
    neighbors = np.tile([0, 1, 2, 3, 4, 5, 6], (10, 1))

    competences, diversity = target.estimate_competence(query, neighbors)
    assert np.allclose(competences, [2. / 7, 4. / 7, 5. / 7])
    assert np.allclose(diversity, [-5. / 7, -4. / 7, -3. / 7])


def test_estimate_competence_Q():
    query = np.ones((1, 2))
    x = np.array([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1])

    clf1 = create_base_classifier(np.array([1, 0, 1, 0, 0, 0, 0]))
    clf2 = create_base_classifier(np.array([1, 0, 0, 0, 1, 0, 0]))
    clf3 = create_base_classifier(np.array([0, 0, 1, 0, 1, 1, 0]))

    pool_classifiers = [clf1, clf2, clf3]

    target = DESKNN(pool_classifiers, k=7, pct_accuracy=1, pct_diversity=1,
                    metric='Q')
    target.fit(x, y)
    neighbors = np.array([[0, 1, 2, 3, 4, 5, 6]])

    competences, diversity = target.estimate_competence(query, neighbors)
    assert np.allclose(competences, [2. / 7, 4. / 7, 5. / 7])
    assert np.allclose(diversity, [2, 1.2, 1.2])


def test_estimate_competence_Q_batch():
    n_samples = 10
    query = np.ones((n_samples, 2))
    x = np.array([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1])

    clf1 = create_base_classifier(np.array([1, 0, 1, 0, 0, 0, 0]))
    clf2 = create_base_classifier(np.array([1, 0, 0, 0, 1, 0, 0]))
    clf3 = create_base_classifier(np.array([0, 0, 1, 0, 1, 1, 0]))

    pool_classifiers = [clf1, clf2, clf3]

    target = DESKNN(pool_classifiers, k=7, pct_accuracy=1, pct_diversity=1,
                    metric='Q')
    target.fit(x, y)
    neighbors = np.tile([0, 1, 2, 3, 4, 5, 6], (n_samples, 1))

    competences, diversity = target.estimate_competence(query, neighbors)
    assert np.allclose(competences, [2. / 7, 4. / 7, 5. / 7])
    assert np.allclose(diversity, [2, 1.2, 1.2])


def test_estimate_competence_ratio():
    query = np.ones((1, 2))

    x = np.array([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1])

    clf1 = create_base_classifier(np.array([1, 0, 1, 0, 0, 0, 0]))
    clf2 = create_base_classifier(np.array([1, 0, 0, 0, 1, 0, 0]))
    clf3 = create_base_classifier(np.array([0, 0, 1, 0, 1, 1, 0]))

    pool_classifiers = [clf1, clf2, clf3]

    target = DESKNN(pool_classifiers, k=7, pct_accuracy=1, pct_diversity=1,
                    metric='ratio')
    target.fit(x, y)
    neighbors = np.array([[0, 1, 2, 3, 4, 5, 6]])

    competences, diversity = target.estimate_competence(query, neighbors)
    assert np.allclose(competences, [2. / 7, 4. / 7, 5. / 7])
    assert np.allclose(diversity, [2.166, 3.666, 4.500], atol=0.01)


def test_estimate_competence_ratio_batch():
    n_samples = 10
    query = np.ones((n_samples, 2))

    x = np.array([0, 1, 2, 3, 4, 5, 6]).reshape(-1, 1)
    y = np.array([0, 0, 0, 0, 1, 1, 1])

    clf1 = create_base_classifier(np.array([1, 0, 1, 0, 0, 0, 0]))
    clf2 = create_base_classifier(np.array([1, 0, 0, 0, 1, 0, 0]))
    clf3 = create_base_classifier(np.array([0, 0, 1, 0, 1, 1, 0]))

    pool_classifiers = [clf1, clf2, clf3]

    target = DESKNN(pool_classifiers, k=7, pct_accuracy=1, pct_diversity=1,
                    metric='ratio')
    target.fit(x, y)
    neighbors = np.tile([0, 1, 2, 3, 4, 5, 6], (n_samples, 1))

    competences, diversity = target.estimate_competence(query, neighbors)
    assert np.allclose(competences, [2. / 7, 4. / 7, 5. / 7])
    assert np.allclose(diversity, [2.166, 3.666, 4.500], atol=0.01)


def test_select():
    """
    Test case: 10 base classifiers; select 5 based on accuracy,
    then the 3 most diverse
    accuracies (/10): 4 6 1 2 9 8 7 9 3 2
        (should select indices_ 1, 4, 5, 6, 7)
    diversity:        0 8 0 0 1 6 7 2 0 0
        (should select indices_ 1, 5, 6 as most diverse)

    """
    pool_classifiers = [create_base_classifier(1) for _ in range(10)]

    accuracies = np.array([4, 6, 1, 2, 9, 8, 7, 9, 3, 2]) / 10.
    diversity = np.array([0, 8, 0, 0, 1, 6, 7, 2, 0, 0])
    target = DESKNN(pool_classifiers, k=7, pct_accuracy=5. / 10,
                    pct_diversity=3. / 10)
    target.N_ = 5
    target.J_ = 3

    selected_classifiers = target.select(accuracies, diversity)
    expected = np.array([[1, 5, 6]])

    assert np.array_equal(np.unique(selected_classifiers), np.unique(expected))


def test_select_batch():
    """
    Test case: 10 base classifiers; select 5 based on accuracy,
    then the 3 most diverse.
    accuracies (/10): 4 6 1 2 9 8 7 9 3 2
        (should select indices_ 1, 4, 5, 6, 7)
    diversity:        0 8 0 0 1 6 7 2 0 0
        (should select indices_ 1, 5, 6 as most diverse)

    """
    n_samples = 10
    pool_classifiers = [create_base_classifier(1) for _ in range(10)]

    accuracies = np.tile([4, 6, 1, 2, 9, 8, 7, 9, 3, 2], (n_samples, 1)) / 10.
    diversity = np.tile([0, 8, 0, 0, 1, 6, 7, 2, 0, 0], (n_samples, 1))
    target = DESKNN(pool_classifiers, k=7, pct_accuracy=5. / 10,
                    pct_diversity=3. / 10)
    target.N_ = 5
    target.J_ = 3

    selected_classifiers = target.select(accuracies, diversity)
    expected = np.tile([1, 5, 6], (n_samples, 1))

    assert np.array_equal(np.unique(selected_classifiers), np.unique(expected))


def test_select_less_diverse():
    """
    Test case: 10 base classifiers; select 5 based on accuracy,
    then the 3 less diverse
    accuracies (/10): 4 6 1 2 9 8 7 9 3 2
        (should select indices_ 1, 4, 5, 6, 7)
    diversity:        0 8 0 0 1 6 7 2 0 0
        (should select indices_ 4, 5, 7 as most diverse)

    """
    pool_classifiers = [create_base_classifier(1) for _ in range(10)]

    accuracies = np.array([[4, 6, 1, 2, 9, 8, 7, 9, 3, 2]]) / 10.
    diversity = np.array([[0, 8, 0, 0, 1, 6, 7, 2, 0, 0]])
    target = DESKNN(pool_classifiers, k=7, pct_accuracy=5. / 10,
                    pct_diversity=3. / 10, more_diverse=False)
    target.N_ = 5
    target.J_ = 3

    selected_classifiers = target.select(accuracies, diversity)
    expected = np.array([[4, 5, 7]])

    assert np.array_equal(np.unique(selected_classifiers), np.unique(expected))


def test_input_diversity_parameter():
    X = np.random.rand(10, 2)
    y = np.ones(10)
    with pytest.raises(ValueError):
        desknn = DESKNN([create_base_classifier(1)] * 100, metric='abc')
        desknn.fit(X, y)


def test_J_N_values():
    X = np.random.rand(10, 2)
    y = np.ones(10)
    with pytest.raises(ValueError):
        desknn = DESKNN([create_base_classifier(1)] * 100, pct_accuracy=0.5,
                        pct_diversity=0)
        desknn.fit(X, y)


def test_J_higher_than_N():
    X = np.random.rand(10, 2)
    y = np.ones(10)
    with pytest.raises(ValueError):
        desknn = DESKNN([create_base_classifier(1)] * 100, pct_accuracy=0.3,
                        pct_diversity=0.5)
        desknn.fit(X, y)


def create_base_classifier(value):
    classifier = MagicMock()
    classifier.predict.return_value = value
    classifier.predict_proba.return_value = value

    return classifier


# Test if the class is raising an error when the base classifiers do not
# implements the predict_proba method.
# In this case the test should not raise an error since this class does not
# require base classifiers that can estimate probabilities
def test_predict_proba():
    X = np.random.randn(15, 5)
    y = np.array([0, 1, 0, 0, 0] * 3)
    clf1 = Perceptron()
    clf1.fit(X, y)
    DESKNN([clf1, clf1, clf1]).fit(X, y)


def test_classify_with_ds_single_sample():
    query = np.ones(2)
    predictions = np.array([0, 1, 0])

    clf1 = create_base_classifier(np.array([1, 0, 1, 0, 0, 0, 0]))
    clf2 = create_base_classifier(np.array([1, 0, 0, 0, 1, 0, 0]))
    clf3 = create_base_classifier(np.array([0, 0, 1, 0, 1, 1, 0]))

    pool_classifiers = [clf1, clf2, clf3]

    desknn_test = DESKNN(pool_classifiers)
    desknn_test.estimate_competence = MagicMock(
        return_value=(np.ones(3), np.ones(3)))
    desknn_test.select = MagicMock(return_value=np.array([[0, 2]]))
    result = desknn_test.classify_with_ds(query, predictions)
    assert np.allclose(result, 0)


def test_classify_with_ds_diff_sizes():
    query = np.ones((10, 2))
    predictions = np.ones((5, 3))

    clf1 = create_base_classifier(np.array([1, 0, 1, 0, 0, 0, 0]))
    clf2 = create_base_classifier(np.array([1, 0, 0, 0, 1, 0, 0]))
    clf3 = create_base_classifier(np.array([0, 0, 1, 0, 1, 1, 0]))

    pool_classifiers = [clf1, clf2, clf3]

    desknn_test = DESKNN(pool_classifiers)

    with pytest.raises(ValueError):
        desknn_test.classify_with_ds(query, predictions)


def test_proba_with_ds_diff_sizes():
    query = np.ones((10, 2))
    predictions = np.ones((5, 3))
    probabilities = np.ones((5, 3, 2))

    clf1 = create_base_classifier(np.array([1, 0, 1, 0, 0, 0, 0]))
    clf2 = create_base_classifier(np.array([1, 0, 0, 0, 1, 0, 0]))
    clf3 = create_base_classifier(np.array([0, 0, 1, 0, 1, 1, 0]))

    pool_classifiers = [clf1, clf2, clf3]

    desknn_test = DESKNN(pool_classifiers)

    with pytest.raises(ValueError):
        desknn_test.predict_proba_with_ds(query, predictions, probabilities)
