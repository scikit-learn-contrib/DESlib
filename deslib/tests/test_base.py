import unittest.mock
from unittest.mock import Mock, MagicMock

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier

from deslib.base import BaseDS
from .conftest import create_base_classifier


def test_all_classifiers_agree():
    # 10 classifiers that return 1
    predictions = np.ones((1, 10))

    assert np.all(BaseDS._all_classifier_agree(predictions))


def test_not_all_classifiers_agree():
    # 10 classifiers that return 1, and one that returns 2
    predictions = np.ones((10, 11))
    predictions[:, -1] = 2

    assert not np.all(BaseDS._all_classifier_agree(predictions))


@pytest.mark.parametrize('query', [None, [np.nan, 1.0]])
def test_predict_value(query):
    pool_classifiers = create_classifiers_disagree()
    ds = BaseDS(pool_classifiers)
    X = np.random.rand(10, 2)
    y = np.ones(10)
    y[:5] = 0
    ds.fit(X, y)
    with pytest.raises(ValueError):
        ds.predict(query)


@pytest.mark.parametrize('k', [0, 1, -1])
def test_check_k_value(k):
    X = np.random.rand(10, 2)
    y = np.ones(10)
    with pytest.raises(ValueError):
        ds_test = BaseDS(k=k)
        ds_test.fit(X, y)


@pytest.mark.parametrize('k', ['a', 2.5])
def test_check_k_type(k):
    X = np.random.rand(10, 2)
    y = np.ones(10)

    with pytest.raises(TypeError):
        ds_test = BaseDS(k=k)
        ds_test.fit(X, y)


@pytest.mark.parametrize('safe_k', ['a', 2.5])
def test_check_safe_k_type(safe_k):
    X = np.random.rand(10, 2)
    y = np.ones(10)
    with pytest.raises(TypeError):
        ds_test = BaseDS(safe_k=safe_k)
        ds_test.fit(X, y)


@pytest.mark.parametrize('safe_k', [0, 1, -1])
def test_check_safe_k_value(safe_k):
    X = np.random.rand(10, 2)
    y = np.ones(10)
    with pytest.raises(ValueError):
        ds_test = BaseDS(safe_k=safe_k)
        ds_test.fit(X, y)


@pytest.mark.parametrize('k, safe_k', [(2, 3), (5, 7)])
def test_valid_safe_k(k, safe_k):
    X = np.random.rand(10, 2)
    y = np.ones(10)
    with pytest.raises(ValueError):
        ds = BaseDS(k=k, safe_k=safe_k)
        ds.fit(X, y)


def create_classifiers_disagree():
    clf1 = create_base_classifier(return_value=1)
    clf_2 = create_base_classifier(return_value=0)
    return [clf1, clf_2]


@pytest.mark.parametrize('knn_method,', ['invalidmethod', 1])
def test_valid_selection_mode(knn_method, create_X_y):
    X, y = create_X_y
    with pytest.raises(ValueError):
        ds = BaseDS(knn_classifier=knn_method)
        ds.fit(X, y)


def test_import_faiss_mode():
    try:
        import sys
        sys.modules.pop('deslib.util.faiss_knn_wrapper')
    except Exception:
        pass
    with unittest.mock.patch.dict('sys.modules', {'faiss': None}):
        with pytest.raises(ImportError):
            BaseDS(knn_classifier="faiss")


def test_none_selection_mode(create_X_y):
    X, y = create_X_y
    ds = BaseDS(knn_classifier=None)
    ds.fit(X, y)
    assert (isinstance(ds.roc_algorithm_, KNeighborsClassifier))


def test_string_selection_mode(create_X_y):
    X, y = create_X_y
    ds = BaseDS(knn_classifier="knn")
    ds.fit(X, y)
    assert (isinstance(ds.roc_algorithm_, KNeighborsClassifier))


# In this test the system was trained for a sample containing 2 features and
# we are passing a sample with 3 as argument. So it should raise a value error.
def test_different_input_shape(create_X_y):
    X, y = create_X_y
    query = np.array([[1.0, 1.0, 2.0]])
    ds_test = BaseDS()
    ds_test.fit(X, y)
    with pytest.raises(ValueError):
        ds_test.predict(query)


def test_empty_pool(create_X_y):
    pool_classifiers = []
    X, y = create_X_y

    with pytest.raises(ValueError):
        ds = BaseDS(pool_classifiers)
        ds.fit(X, y)


# Should raise a NotFittedError since the function 'fit' was
# not called before predict
def test_not_fitted_ds():
    query = np.array([[1.0, 1.0]])

    ds_test = BaseDS()
    with pytest.raises(NotFittedError):
        ds_test.predict(query)


# X has 15 samples while y have 20 labels. So this test should raise an error
def test_input_shape_fit():
    X = np.ones((15, 2))
    y = np.ones(20)
    ds_test = BaseDS()
    with pytest.raises(ValueError):
        ds_test.fit(X, y)


# --------------------Test routines for the DFP (fire DS)--------------------

# Since no classifier crosses the region of competence,
# all of them must be selected
def test_frienemy_no_classifier_crosses(example_estimate_competence,
                                        create_pool_classifiers):
    X, y, neighbors = example_estimate_competence[0:3]

    ds_test = BaseDS(create_pool_classifiers)
    ds_test.fit(X, y)
    mask = ds_test._frienemy_pruning(neighbors[0, :])
    assert mask.shape == (1, 3) and np.allclose(mask, 1)


# In this example, all base classifier should be considered crossing the
# region of competence since they always # predicts the correct label for
# the samples in DSEL.
@pytest.mark.parametrize('index', [0, 1, 2])
def test_frienemy_all_classifiers_crosses(index,
                                          example_all_ones,
                                          create_pool_classifiers):
    X, y, neighbors, _, dsel_processed, _ = example_all_ones
    ds_test = BaseDS(create_pool_classifiers)
    ds_test.fit(X, y)
    ds_test.DSEL_processed_ = dsel_processed

    result = ds_test._frienemy_pruning(neighbors[index, :])
    assert result.all() == 1.0


# Check if the batch processing is working by passing multiple
# samples at the same time.
def test_frienemy_not_all_classifiers_crosses(example_estimate_competence,
                                              create_pool_classifiers):
    expected = np.array([[1, 1, 0], [0, 1, 0], [1, 1, 1]])
    X, y, neighbors, _, dsel_processed, _ = example_estimate_competence
    ds_test = BaseDS(create_pool_classifiers, safe_k=3)
    ds_test.fit(X, y)

    ds_test.DSEL_processed_ = dsel_processed

    # passing three samples to compute the DFP at the same time
    result = ds_test._frienemy_pruning(neighbors)
    assert np.array_equal(result, expected)


# Test the case where the sample is located in a safe region
# (i.e., all neighbors comes from the same class)
def test_frienemy_safe_region(example_estimate_competence,
                              create_pool_classifiers):
    X, y, _, _, dsel_processed, _ = example_estimate_competence
    ds_test = BaseDS(create_pool_classifiers, safe_k=3)
    ds_test.fit(X, y)
    ds_test.DSEL_processed_ = dsel_processed

    result = ds_test._frienemy_pruning(np.array([0, 1, 2, 6, 7, 8, 14]))
    assert np.array_equal(result, np.array([[1, 1, 1]]))


# Check if the batch processing is working by passing multiple samples
# at the same time. Testing sample in a safe region
def test_frienemy_safe_region_batch(example_estimate_competence,
                                    create_pool_classifiers):
    X, y, neighbors, _, dsel_processed, _ = example_estimate_competence

    n_samples = 10
    n_classifiers = 3
    expected = np.ones((n_samples, n_classifiers))
    ds_test = BaseDS(create_pool_classifiers, safe_k=3)
    ds_test.fit(X, y)

    ds_test.DSEL_processed_ = dsel_processed

    neighbors = np.tile(np.array([0, 1, 2, 6, 7, 8, 14]), (n_samples, 1))
    result = ds_test._frienemy_pruning(neighbors)

    assert np.array_equal(result, expected)


@pytest.mark.parametrize('X_test', [None, [[0.1, 0.2], [0.5, np.nan]]])
def test_bad_input_X(X_test, create_X_y):
    X_train, y_train = create_X_y
    ds_test = BaseDS()
    ds_test.fit(X_train, y_train)
    with pytest.raises(ValueError):
        ds_test.predict(X_test)


def test_preprocess_dsel_scores(create_X_y, create_pool_classifiers):
    X, y = create_X_y
    ds_test = BaseDS(create_pool_classifiers)
    ds_test.fit(X, y)
    dsel_scores = ds_test._preprocess_dsel_scores()
    expected = np.array([[0.5, 0.5], [1.0, 0.0], [0.33, 0.67]])
    expected = np.tile(expected, (15, 1, 1))
    assert np.array_equal(dsel_scores, expected)


def test_DFP_is_used(example_estimate_competence, create_pool_classifiers):
    X, y, neighbors, _, dsel_processed, _ = example_estimate_competence
    ds_test = BaseDS(create_pool_classifiers, DFP=True, safe_k=3)
    ds_test.fit(X, y)
    ds_test.DSEL_processed_ = dsel_processed

    DFP_mask = ds_test._frienemy_pruning(neighbors[0, :])
    assert np.array_equal(DFP_mask, np.atleast_2d([1, 1, 0]))


def test_IH_is_used(example_estimate_competence, create_pool_classifiers):
    X, y, neighbors, distances, dsel_processed, _ = example_estimate_competence
    expected = [0, 0, 1]
    query = np.ones((3, 2))
    ds_test = BaseDS(create_pool_classifiers, with_IH=True, IH_rate=0.5)
    ds_test.fit(X, y)

    ds_test.DSEL_processed_ = dsel_processed

    ds_test._get_region_competence = MagicMock(return_value=(distances,
                                                             neighbors))
    predicted = ds_test.predict(query)

    assert np.array_equal(predicted, expected)


@pytest.mark.parametrize('IH_rate', [None, -1, 'abc', 0.75, 1])
def test_input_IH_rate(IH_rate):
    X = np.random.rand(10, 2)
    y = np.ones(10)
    with pytest.raises((ValueError, TypeError)):
        ds = BaseDS(with_IH=True, IH_rate=IH_rate)
        ds.fit(X, y)


def test_predict_proba_all_agree(example_estimate_competence,
                                 create_pool_classifiers):
    X, y, _, _, _, dsel_scores = example_estimate_competence

    query = np.atleast_2d([1, 1])
    ds_test = BaseDS(create_pool_classifiers)
    ds_test.fit(X, y)
    ds_test.DSEL_scores = dsel_scores
    backup_all_agree = BaseDS._all_classifier_agree
    BaseDS._all_classifier_agree = MagicMock(return_value=np.array([True]))
    proba = ds_test.predict_proba(query)

    BaseDS._all_classifier_agree = backup_all_agree
    assert np.allclose(proba, np.atleast_2d([0.61, 0.39]))


# In this test, the three neighborhoods have an hardness level lower
# than the parameter IH_rate (0.5). Thus, the KNN
# Should be used to predict probabilities
@pytest.mark.parametrize('index', [0, 1, 2])
def test_predict_proba_IH_knn(index,
                              example_estimate_competence,
                              create_pool_classifiers):
    X, y, neighbors, distances, _, dsel_scores = example_estimate_competence
    query = np.atleast_2d([1, 1])
    ds_test = BaseDS(create_pool_classifiers, with_IH=True, IH_rate=0.5)
    ds_test.fit(X, y)
    ds_test.DSEL_scores = dsel_scores

    neighbors = neighbors[index, :]
    distances = distances[index, :]

    ds_test._get_region_competence = MagicMock(return_value=(distances,
                                                             neighbors))

    ds_test.roc_algorithm_.predict_proba = MagicMock(
        return_value=np.atleast_2d([0.45, 0.55]))
    proba = ds_test.predict_proba(query)
    assert np.array_equal(proba, np.atleast_2d([0.45, 0.55]))


# In this test, the three neighborhoods have an hardness level higher
# than the parameter IH_rate. Thus, the prediction
# should be passed down to the predict_proba_with_ds function.
@pytest.mark.parametrize('index', [0, 1, 2])
def test_predict_proba_instance_called(index,
                                       example_estimate_competence,
                                       create_pool_classifiers):
    X, y, neighbors, distances, _, _ = example_estimate_competence
    query = np.atleast_2d([1, 1])
    ds_test = BaseDS(create_pool_classifiers, with_IH=True, IH_rate=0.10)
    ds_test.fit(X, y)

    neighbors = neighbors[index, :]
    distances = distances[index, :]

    ds_test._get_region_competence = MagicMock(return_value=(distances,
                                                             neighbors))

    ds_test.predict_proba_with_ds = MagicMock(
        return_value=np.atleast_2d([0.25, 0.75]))
    proba = ds_test.predict_proba(query)
    assert np.allclose(proba, np.atleast_2d([0.25, 0.75]))


# ------------------------ Testing label encoder------------------------
def create_pool_classifiers_dog_cat_plane():
    clf_0 = create_base_classifier(return_value='cat',
                                   return_prob=np.atleast_2d([0.5, 0.5]))

    clf_1 = create_base_classifier(return_value='dog',
                                   return_prob=np.atleast_2d([1.0, 0.0]))

    clf_2 = create_base_classifier(return_value='plane',
                                   return_prob=np.atleast_2d([0.33, 0.67]))

    pool_classifiers = [clf_0, clf_1, clf_2]
    return pool_classifiers


def create_pool_classifiers_dog():
    clf_0 = create_base_classifier(return_value='dog',
                                   return_prob=np.atleast_2d([0.5, 0.5]))
    pool_classifiers = [clf_0, clf_0, clf_0]
    return pool_classifiers


def test_label_encoder_only_dsel_allagree():
    X_dsel_ex1 = np.array([[-1, 1], [-0.75, 0.5], [-1.5, 1.5]])
    y_dsel_ex1 = np.array(['cat', 'dog', 'plane'])

    query = np.atleast_2d([[1, 0], [-1, -1]])
    ds_test = BaseDS(create_pool_classifiers_dog(), k=2)
    ds_test.fit(X_dsel_ex1, y_dsel_ex1)

    predictions = ds_test.predict(query)
    assert np.array_equal(predictions, ['dog', 'dog'])


def test_label_encoder_only_dsel():
    X_dsel_ex1 = np.array([[-1, 1], [-0.75, 0.5], [-1.5, 1.5]])
    y_dsel_ex1 = np.array(['cat', 'dog', 'plane'])

    query = np.atleast_2d([[1, 0], [-1, -1]])
    ds_test = BaseDS(create_pool_classifiers_dog_cat_plane(), k=2)
    ds_test.fit(X_dsel_ex1, y_dsel_ex1)

    ds_test.classify_with_ds = Mock()
    ds_test.classify_with_ds.return_value = [1, 0]
    predictions = ds_test.predict(query)
    assert np.array_equal(predictions, ['dog', 'cat'])


def test_label_encoder_base():
    from sklearn.linear_model import LogisticRegression

    X_dsel_ex1 = np.array([[-1, 1], [-0.75, 0.5], [-1.5, 1.5]])
    y_dsel_ex1 = np.array(['cat', 'dog', 'plane'])

    x = [[-2, -2], [2, 2]]
    y = ['cat', 'dog']
    pool_classifiers = [LogisticRegression().fit(x, y) for _ in range(2)]

    query = np.atleast_2d([[1, 0], [-1, -1]])
    ds_test = BaseDS(pool_classifiers, k=2)
    ds_test.fit(X_dsel_ex1, y_dsel_ex1)
    predictions = ds_test.predict(query)

    assert np.array_equal(predictions, np.array(['dog', 'cat']))

