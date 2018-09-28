import pytest
from sklearn.linear_model import Perceptron

from deslib.des.des_mi import DESMI
from deslib.tests.examples_test import *
from sklearn.utils.estimator_checks import check_estimator


def test_check_estimator():
    check_estimator(DESMI)


# TODO: create test routine for the estimate_competence method


@pytest.mark.parametrize('alpha', [-1.0, -0.5, 0.0])
def test_check_alpha_value(alpha):
    pool_classifiers = create_pool_classifiers()

    with pytest.raises(ValueError):
        DESMI(pool_classifiers, alpha=alpha)


@pytest.mark.parametrize('alpha', ['a', None, 'string', 1])
def test_check_alpha_type(alpha):
    pool_classifiers = create_pool_classifiers()
    with pytest.raises(TypeError):
        desmi = DESMI(pool_classifiers, alpha=alpha)
        desmi.fit(X_dsel_ex1, y_dsel_ex1)


@pytest.mark.parametrize('pct_accuracy', [-1.0, -0.5, 0.0, 1.01])
def test_check_alpha_value(pct_accuracy):
    pool_classifiers = create_pool_classifiers()
    with pytest.raises(ValueError):
        desmi = DESMI(pool_classifiers, pct_accuracy=pct_accuracy)
        desmi.fit(X_dsel_ex1, y_dsel_ex1)


# Test if the class is raising an error when the base classifiers do not implements the predict_proba method.
# In this case the test should not raise an error since this class does not require base classifiers that
# can estimate probabilities
def test_require_proba():
    X = np.random.randn(5, 5)
    y = np.array([0, 1, 0, 0, 0])
    clf1 = Perceptron()
    clf1.fit(X, y)
    DESMI([clf1, clf1, clf1])


def test_select_single_sample():
    pool_classifiers = create_pool_classifiers()
    des_mi = DESMI(pool_classifiers, pct_accuracy=0.7)
    des_mi.N_ = 2
    competences = np.array([0.7, 0.2, 1.0])
    selected_clf = des_mi.select(competences)
    expected = np.array([0, 2])
    assert np.array_equal(np.unique(selected_clf), np.unique(expected))


def test_select_batch_samples():
    n_samples = 10
    pool_classifiers = create_pool_classifiers()
    des_mi = DESMI(pool_classifiers, pct_accuracy=0.7)
    des_mi.N_ = 2
    competences = np.tile(np.array([0.7, 0.2, 1.0]), (n_samples, 1))
    selected_clf = des_mi.select(competences)
    expected = np.tile(np.array([0, 2]), (n_samples, 1))
    assert np.array_equal(np.unique(selected_clf), np.unique(expected))


def test_classify_with_ds_single_sample():
    query = np.ones(2)

    pool_classifiers = create_pool_classifiers()
    # simulated predictions of the pool of classifiers
    predictions = np.array([0, 1, 0])

    desmi_test = DESMI(pool_classifiers, DFP=True)
    desmi_test.DFP_mask = np.ones((1, 3))
    desmi_test.estimate_competence = MagicMock(return_value=(np.ones((1, 3))))
    desmi_test.select = MagicMock(return_value=np.array([[0, 2]]))
    result = desmi_test.classify_with_ds(query, predictions)
    assert np.allclose(result, 0)


def test_classify_with_ds_batch_samples():
    n_samples = 10
    # Passing 10 samples for classification automatically
    query = np.ones((n_samples, 2))

    pool_classifiers = create_pool_classifiers()
    # simulated predictions of the pool of classifiers
    predictions = np.tile(np.array([0, 1, 0]), (n_samples, 1))

    desmi_test = DESMI(pool_classifiers)
    desmi_test.estimate_competence = MagicMock(return_value=(np.ones((n_samples, 3))))
    desmi_test.select = MagicMock(return_value=np.tile(np.array([[0, 2]]), (n_samples, 1)))
    result = desmi_test.classify_with_ds(query, predictions)
    assert np.allclose(result, np.zeros(10))


def test_classify_with_ds_diff_sizes():
    query = np.ones((10, 2))
    predictions = np.ones((5, 3))

    pool_classifiers = create_pool_classifiers()
    des_mi = DESMI(pool_classifiers)

    with pytest.raises(ValueError):
        des_mi.classify_with_ds(query, predictions)


def test_proba_with_ds_diff_sizes():
    query = np.ones((10, 2))
    predictions = np.ones((5, 3))
    probabilities = np.ones((5, 3, 2))

    pool_classifiers = create_pool_classifiers()
    des_mi = DESMI(pool_classifiers)

    with pytest.raises(ValueError):
        des_mi.predict_proba_with_ds(query, predictions, probabilities)


def test_predict_proba_with_ds():
    query = np.array([-1, 1])
    pool_classifiers = create_pool_classifiers() + create_pool_classifiers()
    desmi_test = DESMI(pool_classifiers, DFP=True)
    desmi_test.DFP_mask = np.ones((1, 6))
    selected_indices = np.array([[0, 1, 5]])

    desmi_test.estimate_competence = MagicMock(return_value=np.ones(6))
    desmi_test.select = MagicMock(return_value=selected_indices)

    desmi_test.n_classes = 2
    expected = np.array([0.61, 0.39])

    predictions = []
    probabilities = []
    for clf in desmi_test.pool_classifiers:
        predictions.append(clf.predict(query)[0])
        probabilities.append(clf.predict_proba(query)[0])

    query = np.atleast_2d(query)
    predictions = np.atleast_2d(predictions)
    probabilities = np.array(probabilities)
    probabilities = np.expand_dims(probabilities, axis=0)

    predicted_proba = desmi_test.predict_proba_with_ds(query, predictions, probabilities)
    assert np.isclose(predicted_proba, expected, atol=0.01).all()
