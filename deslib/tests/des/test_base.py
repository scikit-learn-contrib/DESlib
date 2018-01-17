from unittest.mock import MagicMock

import numpy as np
import pytest

from deslib.des.base import DES
from deslib.tests.examples_test import create_pool_classifiers


@pytest.mark.parametrize('mode,', ['a', 'test'])
def test_valid_selection_mode(mode):
    with pytest.raises(ValueError):
        DES(create_pool_classifiers(), mode=mode)


@pytest.mark.parametrize('mode,', [1, [1.0, 2.0], None])
def test_selection_method_type(mode):
    with pytest.raises(TypeError):
        DES(create_pool_classifiers(), mode=mode)


# ------------------------ Testing classify_instance -----------------
"""Example considering a pool composed of 6 base classifiers. The classifiers with index 0, 2, 3 and 5 predicts class 0
while classifiers with indices 1 and 4 predicts class 1.
"""


# In this first example only dynamic selection is considered.  Since the selected indices are 0, 1 and 5 the expected
# prediction should be 0 (2 votes).
def test_classify_instance_selection():
    query = np.array([-1, 1])
    pool_classifiers = create_pool_classifiers() + create_pool_classifiers()
    des_test = DES(pool_classifiers, mode='selection')
    # competences = [0.55, 1.0, 0.2, 0.65, 0.75, 0.8]
    selected_index = [0, 1, 5]
    des_test.select = MagicMock(return_value=selected_index)

    predicted_label = des_test.classify_instance(query)
    assert predicted_label == 0.0


# In this example all classifiers are combined, however they are weighted based on the competence level. Even
# though there is four classifiers giving label 0 and only classifiers 2 giving label 1, the prediction should
# be 1 due to the classifiers weights
def test_classify_instance_weighting():
    query = np.array([-1, 1])
    pool_classifiers = create_pool_classifiers() + create_pool_classifiers()
    des_test = DES(pool_classifiers, mode='weighting')
    competences = np.array([0.55, 1.0, 0.2, 0.60, 0.75, 0.3])
    des_test.estimate_competence = MagicMock(return_value=competences)

    predicted_label = des_test.classify_instance(query)
    assert predicted_label == 1.0


# Same example of test_classify_instance_selection, however, since the weights are also used in the hybrid scheme,
# the function should return 1 instead of 0.
def test_classify_instance_hybrid():
    query = np.array([-1, 1])
    pool_classifiers = create_pool_classifiers() + create_pool_classifiers()
    des_test = DES(pool_classifiers, mode='hybrid')
    selected_indices = [0, 1, 5]
    competences = np.array([0.55, 1.0, 0.2, 0.60, 0.75, 0.3])
    des_test.estimate_competence = MagicMock(return_value=competences)
    des_test.select = MagicMock(return_value=selected_indices)

    predicted_label = des_test.classify_instance(query)
    assert predicted_label == 1.0

# ------------------------ Testing predict_proba -----------------


# The prediction of probability here should be an average_rule of the probabilities estimates of the three selected
# base classifiers
def test_predict_proba_selection():
    query = np.array([-1, 1])
    pool_classifiers = create_pool_classifiers() + create_pool_classifiers()
    des_test = DES(pool_classifiers, mode='selection')
    selected_indices = [0, 1, 5]
    des_test.select = MagicMock(return_value=selected_indices)

    des_test.n_classes = 2

    expected = np.array([0.61, 0.39])
    predicted_proba = des_test.predict_proba_instance(query)
    assert np.isclose(predicted_proba, expected, atol=0.01).all()


# The predicted probabilities must also consider the assigned weights of each base classifier
def test_predict_proba_weighting():
    query = np.array([-1, 1])
    pool_classifiers = create_pool_classifiers()
    des_test = DES(pool_classifiers, mode='weighting')
    competences = np.array([0.5, 1.0, 0.2])
    des_test.estimate_competence = MagicMock(return_value=competences)
    des_test.n_classes = 2
    expected = np.array([0.5769, 0.4231])
    predicted_proba = des_test.predict_proba_instance(query)
    assert np.isclose(predicted_proba, expected, atol=0.01).all()


# The predicted probabilities must also consider the assigned weights of each base classifier selected
def test_predict_proba_hybrid():
    query = np.array([-1, 1])
    pool_classifiers = create_pool_classifiers() + create_pool_classifiers()
    des_test = DES(pool_classifiers, mode='hybrid')
    des_test.n_classes = 2

    selected_indices = [0, 1, 5]
    competences = np.array([0.55, 1.0, 0.2, 0.60, 0.75, 0.3])

    expected = np.array([0.5744, 0.4256])

    des_test.estimate_competence = MagicMock(return_value=competences)
    des_test.select = MagicMock(return_value=selected_indices)

    predicted_proba = des_test.predict_proba_instance(query)
    assert np.isclose(predicted_proba, expected, atol=0.01).all()
