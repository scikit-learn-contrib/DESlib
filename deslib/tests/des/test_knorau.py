import pytest

from deslib.des.knora_u import KNORAU
from deslib.tests.examples_test import *
from sklearn.linear_model import Perceptron


@pytest.mark.parametrize('index, expected', [(0, [4.0, 3.0, 4.0]),
                                             (1, [5.0, 2.0, 5.0]),
                                             (2, [2.0, 5.0, 2.0])])
def test_estimate_competence(index, expected):
    query = np.atleast_2d([1, 1])

    knora_u_test = KNORAU(create_pool_classifiers())
    knora_u_test.fit(X_dsel_ex1, y_dsel_ex1)
    knora_u_test.DFP_mask = np.ones(knora_u_test .n_classifiers)
    knora_u_test.neighbors = neighbors_ex1[index, :]
    knora_u_test.distances = distances_ex1[index, :]
    competences = knora_u_test.estimate_competence(query)
    assert np.isclose(competences, expected, atol=0.01).all()


@pytest.mark.parametrize('index, expected', [(0, 0),
                                             (1, 0),
                                             (2, 1)])
def test_classify(index, expected):
    query = np.atleast_2d([1, 1])

    knora_u_test = KNORAU(create_pool_classifiers())
    knora_u_test.fit(X_dsel_ex1, y_dsel_ex1)
    knora_u_test.DFP_mask = np.ones(knora_u_test .n_classifiers)
    knora_u_test.neighbors = neighbors_ex1[index, :]
    knora_u_test.distances = distances_ex1[index, :]

    predictions = []
    for clf in knora_u_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])

    prediction = knora_u_test.classify_instance(query, np.array(predictions))

    assert prediction == expected


@pytest.mark.parametrize('index, expected', [(0, 0),
                                             (1, 0),
                                             (2, 0)])
def test_classify2(index, expected):
    query = np.atleast_2d([1, 1])

    knora_u_test = KNORAU(create_pool_classifiers()+create_pool_all_agree(0, 1))
    knora_u_test.fit(X_dsel_ex1, y_dsel_ex1)
    knora_u_test.DFP_mask = np.ones(knora_u_test .n_classifiers)
    knora_u_test.neighbors = neighbors_ex1[index, :]
    knora_u_test.distances = distances_ex1[index, :]

    predictions = []
    for clf in knora_u_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])

    prediction = knora_u_test.classify_instance(query, np.array(predictions))

    assert prediction == expected


def test_weights_zero():
    query = np.atleast_2d([1, 1])

    knora_u_test = KNORAU(create_pool_classifiers())
    knora_u_test.estimate_competence = MagicMock(return_value=np.zeros(3))

    result = knora_u_test.select(query)
    assert np.array_equal(result, np.array([0, 1, 0]))


# Test if the class is raising an error when the base classifiers do not implements the predict_proba method.
# In this case the test should not raise an error since this class does not require base classifiers that
# can estimate probabilities
def test_predict_proba():
    X = X_dsel_ex1
    y = y_dsel_ex1
    clf1 = Perceptron()
    clf1.fit(X, y)
    KNORAU([clf1, clf1])





