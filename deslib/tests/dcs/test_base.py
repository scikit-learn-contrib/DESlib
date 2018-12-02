from unittest.mock import MagicMock

import numpy as np
import pytest

from deslib.dcs.base import BaseDCS


@pytest.mark.parametrize('selection_method,', ['a', 'test'])
def test_valid_selection_method(selection_method, create_pool_classifiers):
    X = np.random.rand(10, 2)
    y = np.ones(10)
    with pytest.raises(ValueError):
        dcs = BaseDCS(create_pool_classifiers,
                      selection_method=selection_method)
        dcs.fit(X, y)


@pytest.mark.parametrize('selection_method,', [1, [1.0, 2.0], None, np.nan])
def test_selection_method_type(selection_method, create_pool_classifiers):
    X = np.random.rand(10, 2)
    y = np.ones(10)
    with pytest.raises(TypeError):
        dcs = BaseDCS(create_pool_classifiers,
                      selection_method=selection_method)
        dcs.fit(X, y)


@pytest.mark.parametrize('diff_thresh,', ['test', None, [0.1, 0.2]])
def test_valid_diff_threshold_type(diff_thresh, create_pool_classifiers):
    X = np.random.rand(10, 2)
    y = np.ones(10)
    with pytest.raises(TypeError):
        dcs = BaseDCS(create_pool_classifiers, selection_method='diff',
                      diff_thresh=diff_thresh)
        dcs.fit(X, y)


@pytest.mark.parametrize('diff_thresh,', [1.0, -0.15, 0.5, np.nan])
def test_valid_diff_threshold_value(diff_thresh, create_pool_classifiers):
    X = np.random.rand(10, 2)
    y = np.ones(10)
    with pytest.raises(ValueError):
        dcs = BaseDCS(create_pool_classifiers, selection_method='diff',
                      diff_thresh=diff_thresh)
        dcs.fit(X, y)


# ------------- Testing different types of selection routines -----------------
@pytest.mark.parametrize('competences, expected',
                         [([1.0, 0.5, 0.5], 0), ([0.8, 0.9, 1.0], 2),
                          ([0.0, 0.0, 0.15], 2)])
def test_select_best(competences, expected, create_pool_classifiers):
    pool_classifiers = create_pool_classifiers
    dcs_test = BaseDCS(pool_classifiers, selection_method='best')
    selected_clf = dcs_test.select(np.array(competences))
    assert np.allclose(selected_clf, expected)


@pytest.mark.parametrize('competences, expected',
                         [([1.0, 1.0, 0.5], [True, True, False]),
                          ([0.8, 0.9, 0.9], [False, True, True]),
                          ([0.15, 0.15, 0.15], [True, True, True]),
                          ([0.0, 0.0, 0.0], [True, True, True])])
def test_select_all(competences, expected, create_pool_classifiers):
    pool_classifiers = create_pool_classifiers
    dcs_test = BaseDCS(pool_classifiers, selection_method='all')
    selected_clf = dcs_test.select(np.array(competences))
    assert np.allclose(selected_clf, expected)


@pytest.mark.parametrize('competences, expected',
                         [([1.0, 0.5, 0.5], 0), ([0.8, 0.9, 1.0], 2),
                          ([0.0, 0.0, 0.15], 2)])
def test_select_diff(competences, expected):
    rng = np.random.RandomState(123456)
    dcs_test = BaseDCS(selection_method='diff', diff_thresh=0.15,
                       random_state=rng)

    selected_clf = dcs_test.select(np.array(competences))
    assert np.allclose(selected_clf, expected)


@pytest.mark.parametrize('competences, expected',
                         [([0.5, 0.5, 0.5], 1), ([0.8, 0.9, 1.0], 2),
                          ([0.0, 0.10, 0.0], 1)])
def test_select_random(competences, expected):
    rng = np.random.RandomState(123456)
    dcs_test = BaseDCS(selection_method='random', random_state=rng)

    selected_clf = dcs_test.select(np.array(competences))
    assert np.allclose(selected_clf, expected)


def test_select_best_batch():
    competences = np.array(
        [[1.0, 0.5, 0.5], [0.8, 0.9, 1.0], [0.0, 0.0, 0.15]])
    expected = [0, 2, 2]
    dcs_test = BaseDCS(selection_method='best')
    selected_clf = dcs_test.select(competences)
    assert np.array_equal(selected_clf, expected)


def test_select_all_batch():
    competences = np.array(
        [[1.0, 1.0, 0.5], [0.8, 0.9, 0.9], [0.15, 0.15, 0.15],
         [0.0, 0.0, 0.0]])
    expected = np.array(
        [[True, True, False], [False, True, True], [True, True, True],
         [True, True, True]])
    dcs_test = BaseDCS(selection_method='all')
    selected_clf = dcs_test.select(competences)
    assert np.array_equal(selected_clf, expected)


def test_select_diff_batch():
    competences = np.array(
        [[1.0, 0.5, 0.5], [0.8, 0.9, 1.0], [0.0, 0.0, 0.15]])
    expected = np.array([0, 2, 2])
    rng = np.random.RandomState(123456)
    dcs_test = BaseDCS(selection_method='diff', diff_thresh=0.15,
                       random_state=rng)

    selected_clf = dcs_test.select(competences)
    assert np.array_equal(selected_clf, expected)


def test_select_random_batch():
    competences = np.array(
        [[0.5, 0.5, 0.5], [0.8, 0.9, 1.0], [0.0, 0.10, 0.0]])
    expected = np.array([1, 2, 1])
    rng = np.random.RandomState(123456)
    dcs_test = BaseDCS(selection_method='random', random_state=rng)

    selected_clf = dcs_test.select(competences)
    assert np.array_equal(selected_clf, expected)


# --------------- Testing classify_with_ds and predict_proba -----------------


def test_classify_instance(create_pool_classifiers):
    query = np.array([-1, 1])
    n_classifiers = 3

    pool_classifiers = create_pool_classifiers
    dcs_test = BaseDCS(pool_classifiers)
    competences = np.random.rand(n_classifiers)

    dcs_test.estimate_competence = MagicMock(return_value=competences)
    expected = pool_classifiers[np.argmax(competences)].predict(query)[0]

    predictions = []
    for clf in dcs_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
    predicted_label = dcs_test.classify_with_ds(query, np.array(predictions))
    assert predicted_label == expected


def test_classify_instance_batch(create_pool_classifiers):
    n_samples = 3
    n_classifiers = 3
    query = np.ones((n_samples, 2))
    pool_classifiers = create_pool_classifiers
    dcs_test = BaseDCS(pool_classifiers)

    competences = np.random.rand(n_samples, n_classifiers)

    dcs_test.estimate_competence = MagicMock(return_value=competences)
    expected = []
    for ind in range(n_samples):
        expected.append(
            pool_classifiers[np.argmax(competences[ind, :])].predict(query)[0])

    predictions = []
    for clf in dcs_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
    predicted_label = dcs_test.classify_with_ds(query,
                                                np.tile(predictions, (3, 1)))
    assert np.array_equal(predicted_label, expected)


@pytest.mark.parametrize('competences, expected', [([0.6, 0.2, 0.6], 0),
                                                   ([0.5, 0.8, 0.5], 1)])
def test_classify_instance_all(competences, expected, create_pool_classifiers):
    query = np.array([-1, 1])
    pool_classifiers = create_pool_classifiers
    dcs_test = BaseDCS(pool_classifiers, selection_method='all')
    dcs_test.estimate_competence = MagicMock(
        return_value=np.array(competences))

    predictions = []
    for clf in dcs_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
    predicted_label = dcs_test.classify_with_ds(query, np.array(predictions))
    assert predicted_label == expected


def test_classify_instance_all_batch(create_pool_classifiers):
    competences = np.array([[0.6, 0.2, 0.6], [0.5, 0.8, 0.5]])
    expected = [0, 1]
    n_samples = 2
    query = np.ones((n_samples, 2))
    pool_classifiers = create_pool_classifiers
    dcs_test = BaseDCS(pool_classifiers, selection_method='all')
    dcs_test.estimate_competence = MagicMock(
        return_value=np.array(competences))

    predictions = []
    for clf in dcs_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
    predicted_label = dcs_test.classify_with_ds(query, np.tile(predictions,
                                                               (n_samples, 1)))
    assert np.array_equal(predicted_label, expected)


def test_predict_proba_instance(create_pool_classifiers):
    query = np.array([-1, 1])
    n_classifiers = 3
    pool_classifiers = create_pool_classifiers
    dcs_test = BaseDCS(pool_classifiers)
    dcs_test.n_classes_ = 2

    competences = np.random.rand(n_classifiers)

    dcs_test.estimate_competence = MagicMock(return_value=competences)
    expected = pool_classifiers[np.argmax(competences)].predict_proba(query)

    predictions = []
    probabilities = []
    for clf in dcs_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
        probabilities.append(clf.predict_proba(query)[0])

    query = np.atleast_2d(query)
    predictions = np.atleast_2d(predictions)
    probabilities = np.array(probabilities)
    probabilities = np.expand_dims(probabilities, axis=0)

    predicted_proba = dcs_test.predict_proba_with_ds(query, predictions,
                                                     probabilities)
    assert np.array_equal(predicted_proba, expected)


@pytest.mark.parametrize('competences, expected',
                         [([0.6, 0.2, 0.6], [0.415, 0.585]),
                          ([0.5, 0.8, 0.5], [1.0, 0.0])])
def test_predict_proba_instance_all(competences, expected,
                                    create_pool_classifiers):
    query = np.array([-1, 1])
    pool_classifiers = create_pool_classifiers
    dcs_test = BaseDCS(pool_classifiers, selection_method='all')
    dcs_test.n_classes_ = 2

    dcs_test.estimate_competence = MagicMock(
        return_value=np.array(competences))

    predictions = []
    probabilities = []
    for clf in dcs_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
        probabilities.append(clf.predict_proba(query)[0])

    query = np.atleast_2d(query)
    predictions = np.atleast_2d(predictions)
    probabilities = np.array(probabilities)
    probabilities = np.expand_dims(probabilities, axis=0)

    predicted_proba = dcs_test.predict_proba_with_ds(query, predictions,
                                                     probabilities)
    assert np.isclose(predicted_proba, expected).all()


def test_classify_with_ds_diff_sizes():
    query = np.ones((10, 2))
    predictions = np.ones((5, 3))
    dcs_test = BaseDCS()

    with pytest.raises(ValueError):
        dcs_test.classify_with_ds(query, predictions)


def test_proba_with_ds_diff_sizes():
    query = np.ones((10, 2))
    predictions = np.ones((5, 3))
    probabilities = np.ones((5, 3, 2))
    dcs_test = BaseDCS()

    with pytest.raises(ValueError):
        dcs_test.predict_proba_with_ds(query, predictions, probabilities)
