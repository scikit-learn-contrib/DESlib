from unittest.mock import MagicMock

import numpy as np
import pytest

from deslib.des.base import BaseDES


@pytest.mark.parametrize('mode,', ['a', 'test'])
def test_valid_selection_mode(mode, create_pool_classifiers):
    X = np.random.rand(10, 2)
    y = np.ones(10)
    with pytest.raises(ValueError):
        des = BaseDES(create_pool_classifiers, mode=mode)
        des.fit(X, y)


@pytest.mark.parametrize('mode,', [1, [1.0, 2.0], None])
def test_selection_method_type(mode, create_pool_classifiers):
    X = np.random.rand(10, 2)
    y = np.ones(10)
    with pytest.raises(TypeError):
        des = BaseDES(create_pool_classifiers, mode=mode)
        des.fit(X, y)


# ------------------------ Testing classify_with_ds -----------------
"""Example considering a pool composed of 6 base classifiers. The classifiers
with index 0, 2, 3 and 5 predicts class 0
while classifiers with indices_ 1 and 4 predicts class 1.
"""


# In this first example only dynamic selection is considered.  Since the
# selected indices_ are 0, 1 and 5 the expected prediction should be 0
# (2 votes).
def test_classify_instance_selection(create_pool_classifiers):
    query = np.array([-1, 1])
    pool_classifiers = create_pool_classifiers + create_pool_classifiers
    des_test = BaseDES(pool_classifiers, mode='selection')
    selected_index = np.array([[True, True, False, False, False, True]])
    des_test.select = MagicMock(return_value=selected_index)

    predictions = []
    for clf in des_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])

    predicted_label = des_test.classify_with_ds(query, np.array(predictions))
    assert predicted_label == 0.0


# In this first example only dynamic selection is considered.  Since the
# selected indices_ are 0, 1 and 5 the expected
# prediction should be 0 (2 votes).
def test_classify_instance_selection_batch(create_pool_classifiers):
    n_samples = 3
    query = np.ones((n_samples, 2))
    pool_classifiers = create_pool_classifiers + create_pool_classifiers
    des_test = BaseDES(pool_classifiers, mode='selection')
    selected_index = np.array(
        [[True, True, False, False, False, True] * n_samples])
    des_test.select = MagicMock(return_value=selected_index)

    predictions = []
    for clf in des_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])

    predicted_label = des_test.classify_with_ds(query, np.tile(predictions,
                                                               (n_samples, 1)))
    assert np.allclose(predicted_label, 0) and predicted_label.size == 3


# In this example all classifiers are combined, however they are weighted based
# on the competence level. Even though there is four classifiers giving label 0
# and only classifiers 2 giving label 1, the prediction should
# be 1 due to the classifiers weights
def test_classify_instance_weighting(create_pool_classifiers):
    query = np.array([-1, 1])

    pool_classifiers = create_pool_classifiers + create_pool_classifiers
    des_test = BaseDES(pool_classifiers, mode='weighting')
    des_test.classes_ = np.array([0, 1])
    des_test.n_classes_ = 2

    competences = np.array([0.55, 1.0, 0.2, 0.60, 0.75, 0.3])
    des_test.estimate_competence = MagicMock(return_value=competences)

    predictions = []
    for clf in des_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
    predicted_label = des_test.classify_with_ds(query, np.array(predictions))
    assert predicted_label == 1.0


def test_classify_instance_weighting_batch(create_pool_classifiers):
    n_samples = 3
    query = np.ones((n_samples, 2))
    pool_classifiers = create_pool_classifiers + create_pool_classifiers
    des_test = BaseDES(pool_classifiers, mode='weighting')
    des_test.classes_ = np.array([0, 1])
    des_test.n_classes_ = 2

    competences = np.tile([0.55, 1.0, 0.2, 0.60, 0.75, 0.3], (3, 1))
    des_test.estimate_competence = MagicMock(return_value=competences)

    predictions = []
    for clf in des_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
    predicted_label = des_test.classify_with_ds(query,
                                                np.tile(predictions, (3, 1)))
    assert np.allclose(predicted_label, 1) and predicted_label.size == 3


# Same example of test_classify_instance_selection, however, since the weights
# are also used in the hybrid scheme,
# the function should return 1 instead of 0.
def test_classify_instance_hybrid(create_pool_classifiers):
    query = np.array([-1, 1])
    expected = 1

    pool_classifiers = create_pool_classifiers + create_pool_classifiers
    des_test = BaseDES(pool_classifiers, mode='hybrid')
    des_test.classes_ = np.array([0, 1])
    des_test.n_classes_ = 2
    selected_indices = np.array([[True, True, False, False, False, True]])
    competences = np.array([[0.55, 1.0, 0.2, 0.60, 0.75, 0.3]])
    des_test.estimate_competence = MagicMock(return_value=competences)
    des_test.select = MagicMock(return_value=selected_indices)

    predictions = []
    for clf in des_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])

    predicted_label = des_test.classify_with_ds(query, np.array(predictions))
    assert expected == predicted_label


# Same example of test_classify_instance_selection, however, since the weights
# are also used in the hybrid scheme,
# the function should return 1 instead of 0.
def test_classify_instance_hybrid_batch(create_pool_classifiers):
    query = np.ones((3, 2))
    expected = 1
    pool_classifiers = create_pool_classifiers + create_pool_classifiers
    des_test = BaseDES(pool_classifiers, mode='hybrid')
    des_test.classes_ = np.array([0, 1])
    des_test.n_classes_ = 2

    selected_indices = np.tile([True, True, False, False, False, True], (3, 1))
    competences = np.tile([0.55, 1.0, 0.2, 0.60, 0.75, 0.3], (3, 1))
    des_test.estimate_competence = MagicMock(return_value=competences)
    des_test.select = MagicMock(return_value=selected_indices)

    predictions = []
    for clf in des_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])

    predicted_label = des_test.classify_with_ds(query,
                                                np.tile(predictions, (3, 1)))
    assert np.allclose(predicted_label, expected)


# ------------------------ Testing predict_proba -----------------


# The prediction of probability here should be an average_rule of the
# probabilities estimates of the three selected base classifiers
def test_predict_proba_selection(create_pool_classifiers):
    query = np.array([-1, 1])
    pool_classifiers = create_pool_classifiers + create_pool_classifiers
    des_test = BaseDES(pool_classifiers, mode='selection')
    selected_indices = np.array([0, 1, 5])
    selected_classifiers = np.zeros((1, 6), dtype=bool)
    selected_classifiers[0, selected_indices] = 1
    des_test.select = MagicMock(return_value=selected_classifiers)

    des_test.n_classes_ = 2
    expected = np.array([0.61, 0.39])

    predictions = []
    probabilities = []
    for clf in des_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
        probabilities.append(clf.predict_proba(query)[0])

    query = np.atleast_2d(query)
    predictions = np.atleast_2d(predictions)
    probabilities = np.array(probabilities)
    probabilities = np.expand_dims(probabilities, axis=0)

    predicted_proba = des_test.predict_proba_with_ds(query, predictions,
                                                     probabilities)
    assert np.isclose(predicted_proba, expected, atol=0.01).all()


# The predicted probabilities must also consider the assigned weights of each
# base classifier
def test_predict_proba_weighting(create_pool_classifiers):
    query = np.array([-1, 1])
    pool_classifiers = create_pool_classifiers
    des_test = BaseDES(pool_classifiers, mode='weighting')
    competences = np.array([[0.5, 1.0, 0.2]])
    des_test.estimate_competence = MagicMock(return_value=competences)

    des_test.n_classes_ = 2
    expected = np.array([0.5769, 0.4231])

    predictions = []
    probabilities = []
    for clf in des_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
        probabilities.append(clf.predict_proba(query)[0])

    query = np.atleast_2d(query)
    predictions = np.atleast_2d(predictions)
    probabilities = np.array(probabilities)
    probabilities = np.expand_dims(probabilities, axis=0)

    predicted_proba = des_test.predict_proba_with_ds(query, predictions,
                                                     probabilities)
    assert np.isclose(predicted_proba, expected, atol=0.01).all()


# The predicted probabilities must also consider the assigned weights of each
# base classifier selected
def test_predict_proba_hybrid(create_pool_classifiers):
    query = np.array([-1, 1])
    pool_classifiers = create_pool_classifiers + create_pool_classifiers
    des_test = BaseDES(pool_classifiers, mode='hybrid')
    des_test.n_classes_ = 2

    selected_indices = [0, 1, 5]
    competences = np.array([[0.55, 1.0, 0.2, 0.60, 0.75, 0.3]])

    expected = np.array([0.5744, 0.4256])

    des_test.estimate_competence = MagicMock(return_value=competences)

    selected_classifiers = np.zeros((1, 6), dtype=bool)
    selected_classifiers[0, selected_indices] = 1
    des_test.select = MagicMock(return_value=selected_classifiers)

    predictions = []
    probabilities = []
    for clf in des_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
        probabilities.append(clf.predict_proba(query)[0])

    query = np.atleast_2d(query)
    predictions = np.atleast_2d(predictions)
    probabilities = np.array(probabilities)
    probabilities = np.expand_dims(probabilities, axis=0)

    predicted_proba = des_test.predict_proba_with_ds(query, predictions,
                                                     probabilities)
    assert np.isclose(predicted_proba, expected, atol=0.01).all()


def test_classify_with_ds_diff_sizes(create_pool_classifiers):
    query = np.ones((10, 2))
    predictions = np.ones((5, 3))
    des_test = BaseDES(create_pool_classifiers)

    with pytest.raises(ValueError):
        des_test.classify_with_ds(query, predictions)


def test_proba_with_ds_diff_sizes(create_pool_classifiers):
    query = np.ones((10, 2))
    predictions = np.ones((5, 3))
    probabilities = np.ones((5, 3, 2))
    des_test = BaseDES(create_pool_classifiers)

    with pytest.raises(ValueError):
        des_test.predict_proba_with_ds(query, predictions, probabilities)
