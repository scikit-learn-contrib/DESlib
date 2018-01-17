import pytest

from deslib.dcs.base import DCS
from deslib.tests.examples_test import *


@pytest.mark.parametrize('selection_method,', ['a', 'test'])
def test_valid_selection_method(selection_method):
    with pytest.raises(ValueError):
        DCS(create_pool_classifiers(), selection_method=selection_method)


@pytest.mark.parametrize('selection_method,', [1, [1.0, 2.0], None, np.nan])
def test_selection_method_type(selection_method):
    with pytest.raises(TypeError):
        DCS(create_pool_classifiers(), selection_method=selection_method)


@pytest.mark.parametrize('diff_thresh,', ['test', None, [0.1, 0.2]])
def test_valid_diff_threshold_type(diff_thresh):
    with pytest.raises(TypeError):
        DCS(create_pool_classifiers(), selection_method='diff', diff_thresh=diff_thresh)


@pytest.mark.parametrize('diff_thresh,', [1.0, -0.15, 0.5, np.nan])
def test_valid_diff_threshold_value(diff_thresh):
    with pytest.raises(ValueError):
        DCS(create_pool_classifiers(), selection_method='diff', diff_thresh=diff_thresh)


# ------------------------ Testing different types of selection routines -----------------
@pytest.mark.parametrize('competences, expected', [([1.0, 0.5, 0.5], 0), ([0.8, 0.9, 1.0], 2), ([0.0, 0.0, 0.15], 2)])
def test_select_best(competences, expected):
    pool_classifiers = create_pool_classifiers()
    dcs_test = DCS(pool_classifiers, selection_method='best')
    selected_clf = dcs_test.select(competences)
    assert selected_clf == expected


@pytest.mark.parametrize('competences, expected', [([1.0, 1.0, 0.5], [0, 1]),
                                                   ([0.8, 0.9, 0.9], [1, 2]),
                                                   ([0.15, 0.15, 0.15], [0, 1, 2]),
                                                   ([0.0, 0.0, 0.0], [0, 1, 2])])
def test_select_all(competences, expected):
    pool_classifiers = create_pool_classifiers()
    dcs_test = DCS(pool_classifiers, selection_method='all')
    selected_clf = dcs_test.select(competences)
    assert selected_clf == expected


@pytest.mark.parametrize('competences, expected', [([1.0, 0.5, 0.5], 0), ([0.8, 0.9, 1.0], 2), ([0.0, 0.0, 0.15], 2)])
def test_select_diff(competences, expected):
    rng = np.random.RandomState(123456)
    pool_classifiers = create_pool_classifiers()
    dcs_test = DCS(pool_classifiers, selection_method='diff', diff_thresh=0.15, rng=rng)
    selected_clf = dcs_test.select(competences)
    assert selected_clf == expected


@pytest.mark.parametrize('competences, expected', [([0.5, 0.5, 0.5], 1), ([0.8, 0.9, 1.0], 2), ([0.0, 0.10, 0.0], 1)])
def test_select_random(competences, expected):
    rng = np.random.RandomState(123456)
    pool_classifiers = create_pool_classifiers()
    dcs_test = DCS(pool_classifiers, selection_method='random', rng=rng)
    selected_clf = dcs_test.select(competences)
    assert selected_clf == expected

# ------------------------ Testing classify_instance and predict_proba -----------------


def test_classify_instance():
    query = np.array([-1, 1])
    pool_classifiers = create_pool_classifiers()
    dcs_test = DCS(pool_classifiers)
    competences = np.random.rand(dcs_test.n_classifiers)

    dcs_test.estimate_competence = MagicMock(return_value=competences)
    expected = pool_classifiers[np.argmax(competences)].predict(query)[0]

    predicted_label = dcs_test.classify_instance(query)
    assert predicted_label == expected


@pytest.mark.parametrize('competences, expected', [([0.6, 0.2, 0.6], 0),
                                                   ([0.5, 0.8, 0.5], 1)])
def test_classify_instance_all(competences, expected):
    query = np.array([-1, 1])
    pool_classifiers = create_pool_classifiers()
    dcs_test = DCS(pool_classifiers, selection_method='all')
    dcs_test.estimate_competence = MagicMock(return_value=competences)
    predicted_label = dcs_test.classify_instance(query)
    assert predicted_label == expected


def test_predict_proba_instance():
    query = np.array([-1, 1])
    pool_classifiers = create_pool_classifiers()
    dcs_test = DCS(pool_classifiers)
    dcs_test.n_classes = 2

    competences = np.random.rand(dcs_test.n_classifiers)

    dcs_test.estimate_competence = MagicMock(return_value=competences)
    expected = pool_classifiers[np.argmax(competences)].predict_proba(query)

    predicted_proba = dcs_test.predict_proba_instance(query)
    assert np.array_equal(predicted_proba, expected)


@pytest.mark.parametrize('competences, expected', [([0.6, 0.2, 0.6], [0.415, 0.585]),
                                                   ([0.5, 0.8, 0.5], [1.0, 0.0])])
def test_predict_proba_instance_all(competences, expected):
    query = np.array([-1, 1])
    pool_classifiers = create_pool_classifiers()
    dcs_test = DCS(pool_classifiers, selection_method='all')
    dcs_test.n_classes = 2

    dcs_test.estimate_competence = MagicMock(return_value=competences)
    predicted_proba = dcs_test.predict_proba_instance(query)
    assert np.isclose(predicted_proba, expected).all()
