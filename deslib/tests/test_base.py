import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import Perceptron

from deslib.base import DS
from deslib.tests.examples_test import *


def test_all_classifiers_agree():
    # 10 classifiers that return 1
    pool_classifiers = [create_base_classifier(return_value=1)] * 10
    ds = DS(pool_classifiers)

    x = np.ones((1, 10))
    assert ds._all_classifier_agree(x)


def test_not_all_classifiers_agree():
    # 10 classifiers that return 1, and one that returns 2
    pool_classifiers = [create_base_classifier(return_value=1)] * 10
    pool_classifiers.append(create_base_classifier(return_value=2))
    ds = DS(pool_classifiers)

    x = np.ones((1, 10))
    assert not ds._all_classifier_agree(x)


@pytest.mark.parametrize('query', [None, [np.nan, 1.0]])
def test_predict_value(query):
    pool_classifiers = create_classifiers_disagree()
    ds = DS(pool_classifiers)

    with pytest.raises(ValueError):
        ds.predict(query)


@pytest.mark.parametrize('k', [0, 1, -1])
def test_check_k_value(k):
    pool_classifiers = create_pool_classifiers()

    with pytest.raises(ValueError):
        DS(pool_classifiers, k=k)


@pytest.mark.parametrize('k', ['a', 2.5])
def test_check_k_type(k):
    pool_classifiers = create_pool_classifiers()

    with pytest.raises(TypeError):
        DS(pool_classifiers, k=k)


@pytest.mark.parametrize('safe_k', ['a', 2.5])
def test_check_safe_k_type(safe_k):
    pool_classifiers = create_pool_classifiers()

    with pytest.raises(TypeError):
        DS(pool_classifiers, safe_k=safe_k)


@pytest.mark.parametrize('safe_k', [0, 1, -1])
def test_check_safe_k_value(safe_k):
    pool_classifiers = create_pool_classifiers()

    with pytest.raises(ValueError):
        DS(pool_classifiers, safe_k=safe_k)


@pytest.mark.parametrize('k, safe_k', [(2, 3), (5, 7)])
def test_valid_safe_k(k, safe_k):
    with pytest.raises(ValueError):
        DS([create_base_classifier(1)], k=k, safe_k=safe_k)


def create_classifiers_disagree():
    clf1 = create_base_classifier(return_value=1)
    clf_2 = create_base_classifier(return_value=0)
    return [clf1, clf_2]


# In this test the system was trained for a sample containing 2 features and we are passing a sample with 3 as argument.
# So it should raise a value error.
def test_different_input_shape():
    query = np.array([[1.0, 1.0, 2.0]])
    ds_test = DS(create_pool_classifiers())
    ds_test.fit(X_dsel_ex1, y_dsel_ex1)
    with pytest.raises(ValueError):
        ds_test.predict(query)


def test_empty_pool():
    pool_classifiers = []
    with pytest.raises(ValueError):
        DS(pool_classifiers)


# Should raise a NotFittedError since the function 'fit' was not called before predict
def test_not_fitted_ds():
    query = np.array([[1.0, 1.0]])

    ds_test = DS(create_pool_classifiers())
    with pytest.raises(NotFittedError):
        ds_test.predict(query)


# X has 15 samples while y have 20 labels. So this test should raise an error
def test_input_shape_fit():
    X = X_dsel_ex1
    y = np.ones(20)
    ds_test = DS(create_pool_classifiers())
    with pytest.raises(ValueError):
        ds_test.fit(X, y)


# Tests if the base estimators implements the predict proba function
def test_predict_proba_exists():
    query = np.array([[1.0, 1.0]])

    X = X_dsel_ex1
    y = y_dsel_ex1
    clf1 = Perceptron()
    clf1.fit(X, y)
    with pytest.raises(ValueError):
        DS([clf1, clf1])


@pytest.mark.parametrize('index, expected', [(0, 0.42),
                                             (1, 0.28),
                                             (2, 0.28)])
def test_instance_hardness_region(index, expected):
    X = X_dsel_ex1
    y = y_dsel_ex1
    ds_test = DS(create_pool_classifiers())
    ds_test.fit(X, y)
    neighbors = neighbors_ex1[index, :]
    IH = ds_test._hardness_region_competence(neighbors)
    assert np.isclose(IH, expected, atol=0.01)


def test_instance_hardness_region_all_same():
    X = X_dsel_ex1
    y = y_dsel_ex1
    ds_test = DS(create_pool_classifiers())
    ds_test.fit(X, y)
    neighbors = [0, 1, 2, 6, 7, 8, 13]
    IH = ds_test._hardness_region_competence(neighbors)
    assert IH == 0.0


# Since no classifier crosses the region of competence, all of them must be selected
def test_frienemy_no_classifier_crosses():
    X = X_dsel_ex1
    y = y_dsel_ex1
    ds_test = DS(create_pool_classifiers())
    ds_test.fit(X, y)
    ds_test.neighbors = neighbors_ex1[0, :]
    mask = ds_test._frienemy_pruning()
    assert mask.size == 3 and mask.all() == 1


# In this example, all base classifier should be considered crossing the region of competence since they always
# predicts the correct label for the samples in DSEL.
@pytest.mark.parametrize('index', [0, 1, 2])
def test_frienemy_all_classifiers_crosses(index):
    ds_test = DS(create_pool_classifiers())
    ds_test.processed_dsel = dsel_processed_all_ones
    ds_test.DSEL_target = y_dsel_ex1
    ds_test.DSEL_data = X_dsel_ex1
    ds_test.neighbors = neighbors_ex1[index, :]
    result = ds_test._frienemy_pruning()
    assert result.all() == 1.0


def test_frienemy_not_all_classifiers_crosses():
    ds_test = DS(create_pool_classifiers(), safe_k=3)
    ds_test.processed_dsel = dsel_processed_ex1
    ds_test.DSEL_target = y_dsel_ex1
    ds_test.DSEL_data = X_dsel_ex1
    ds_test.neighbors = neighbors_ex1[0, :]
    result = ds_test._frienemy_pruning()
    assert np.array_equal(result, np.array([1, 1, 0]))


def test_frienemy_safe_region():
    ds_test = DS(create_pool_classifiers(), safe_k=3)
    ds_test.processed_dsel = dsel_processed_ex1
    ds_test.DSEL_target = y_dsel_ex1
    ds_test.DSEL_data = X_dsel_ex1
    ds_test.neighbors = np.array([0, 1, 2, 6, 7, 8, 14])
    result = ds_test._frienemy_pruning()
    assert np.array_equal(result, np.array([1, 1, 1]))


@pytest.mark.parametrize('X', [None, [[0.1, 0.2], [0.5, np.nan]]])
def test_bad_input_X(X):
    ds_test = DS(create_pool_classifiers())
    ds_test.fit(X_dsel_ex1, y_dsel_ex1)
    with pytest.raises(ValueError):
        ds_test.predict(X)


def test_output_profiles_transform():
    query = np.array([[1.0, 1.0]])
    ds_test = DS(create_pool_classifiers())
    ds_test.fit(X_dsel_ex1, y_dsel_ex1)
    profile = ds_test._output_profile_transform(query)
    assert np.array_equal(profile, np.array([0.5, 0.5, 1.0, 0.0, 0.33, 0.67]))


def test_preprocess_dsel_scores():
    ds_test = DS(create_pool_classifiers())
    ds_test.fit(X_dsel_ex1, y_dsel_ex1)
    dsel_scores = ds_test._preprocess_dsel_scores()
    expected = np.ones((15, 6)) * np.array([0.5, 0.5, 1.0, 0.0, 0.33, 0.67])
    assert np.array_equal(dsel_scores, expected)


def test_get_dsel_scores():
    ds_test = DS(create_pool_classifiers())
    ds_test.fit(X_dsel_ex1, y_dsel_ex1)
    ds_test.dsel_scores = dsel_scores_ex1
    assert np.array_equal(ds_test._get_scores_dsel(0, 0), np.array([1.0, 0.0]))
    assert np.array_equal(ds_test._get_scores_dsel(1, 0), np.array([0.5, 0.5]))
    assert np.array_equal(ds_test._get_scores_dsel(2, 0), np.array([0.8, 0.2]))


def test_get_dsel_scores_all_samples():
    ds_test = DS(create_pool_classifiers())
    ds_test.fit(X_dsel_ex1, y_dsel_ex1)
    ds_test.dsel_scores = dsel_scores_ex1
    expected = np.ones((15, 2)) * 0.5
    assert np.array_equal(ds_test._get_scores_dsel(1), expected)


def test_get_dsel_scores_not_processed():
    ds_test = DS(create_pool_classifiers())
    ds_test.fit(X_dsel_ex1, y_dsel_ex1)
    with pytest.raises(NotFittedError):
        ds_test._get_scores_dsel(0)


def test_DFP_is_used():
    query = np.atleast_2d([1, 0])
    ds_test = DS(create_pool_classifiers(), DFP=True, safe_k=3)
    ds_test.fit(X_dsel_ex1, y_dsel_ex1)
    ds_test.processed_dsel = dsel_processed_ex1
    ds_test.DSEL_target = y_dsel_ex1
    ds_test.DSEL_data = X_dsel_ex1
    ds_test.neighbors = neighbors_ex1[0, :]
    ds_test.distances = distances_ex1[0, :]
    ds_test.classify_instance = MagicMock(return_value=0)
    ds_test.predict(query)
    assert np.array_equal(ds_test.DFP_mask, np.array([1, 1, 0]))


@pytest.mark.parametrize('index, expected', [(0, 0), (1, 0), (2, 1)])
def test_IH_is_used(index, expected):
    query = np.atleast_2d([1, 0])
    ds_test = DS(create_pool_classifiers(), with_IH=True, IH_rate=0.5)
    ds_test.fit(X_dsel_ex1, y_dsel_ex1)

    ds_test.processed_dsel = dsel_processed_ex1
    ds_test.DSEL_target = y_dsel_ex1
    ds_test.DSEL_data = X_dsel_ex1

    ds_test.neighbors = neighbors_ex1[index, :]
    ds_test.distances = distances_ex1[index, :]

    predicted = ds_test.predict(query)

    assert predicted == expected


@pytest.mark.parametrize('IH_rate', [None, -1, 'abc', 0.75, 1])
def test_input_IH_rate(IH_rate):

    with pytest.raises((ValueError, TypeError)):
        DS(create_pool_classifiers(), with_IH=True, IH_rate=IH_rate)


def test_predict_proba_all_agree():
    query = np.atleast_2d([1, 1])
    ds_test = DS(create_pool_classifiers())
    ds_test.fit(X_dsel_ex1, y_dsel_ex1)
    ds_test.dsel_scores = dsel_scores_ex1
    ds_test._all_classifier_agree = MagicMock(return_value=True)
    proba = ds_test.predict_proba(query)
    assert np.isclose(proba, np.atleast_2d([0.61, 0.39])).all()


# In this test, the three neighborhoods have an hardness level lower than the parameter IH_rate (0.5). Thus, the KNN
# Should be used to predict probabilities
@pytest.mark.parametrize('index', [0, 1, 2])
def test_predict_proba_IH_knn(index):
    query = np.atleast_2d([1, 1])
    ds_test = DS(create_pool_classifiers(), with_IH=True, IH_rate=0.5)
    ds_test.fit(X_dsel_ex1, y_dsel_ex1)
    ds_test.dsel_scores = dsel_scores_ex1

    ds_test.neighbors = neighbors_ex1[index, :]
    ds_test.distances = distances_ex1[index, :]

    ds_test.roc_algorithm.predict_proba = MagicMock(return_value=np.atleast_2d([0.45, 0.55]))
    proba = ds_test.predict_proba(query)
    assert np.isclose(proba, np.atleast_2d([0.45, 0.55])).all()


# In this test, the three neighborhoods have an hardness level higher than the parameter IH_rate. Thus, the prediction
# should be passed down to the predict_proba_instance function.
@pytest.mark.parametrize('index', [0, 1, 2])
def test_predict_proba_instance_called(index):
    query = np.atleast_2d([1, 1])
    ds_test = DS(create_pool_classifiers(), with_IH=True, IH_rate=0.10)
    ds_test.fit(X_dsel_ex1, y_dsel_ex1)

    ds_test.neighbors = neighbors_ex1[index, :]
    ds_test.distances = distances_ex1[index, :]

    ds_test.predict_proba_instance = MagicMock(return_value=np.atleast_2d([0.25, 0.75]))
    proba = ds_test.predict_proba(query)
    assert np.isclose(proba, np.atleast_2d([0.25, 0.75])).all()


# In this test, the frienemy pruning is used. So, the value of self.DFP_mask should change.
def test_predict_proba_DFP():
    query = np.atleast_2d([1, 1])
    ds_test = DS(create_pool_classifiers(), DFP=True, safe_k=3)
    ds_test.fit(X_dsel_ex1, y_dsel_ex1)

    # change the state of the system
    ds_test.processed_dsel = dsel_processed_ex1
    ds_test.DSEL_target = y_dsel_ex1
    ds_test.DSEL_data = X_dsel_ex1
    ds_test.neighbors = neighbors_ex1[0, :]
    ds_test.distances = distances_ex1[0, :]

    ds_test.predict_proba_instance = MagicMock(return_value=np.atleast_2d([0.25, 0.75]))
    ds_test.predict_proba(query)
    assert np.array_equal(ds_test.DFP_mask, np.array([1, 1, 0]))

