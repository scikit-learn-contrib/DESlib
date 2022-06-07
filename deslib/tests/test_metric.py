import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from deslib.base import BaseDS
# DCS techniques
from deslib.dcs.a_posteriori import APosteriori
from deslib.dcs.mcb import MCB
from deslib.dcs.ola import OLA
from deslib.dcs.rank import Rank
from deslib.des import DESKL
# DES techniques
from deslib.des.des_knn import DESKNN
from deslib.des.des_p import DESP
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU
from deslib.des.meta_des import METADES


# ----- Unit tests -----

@pytest.mark.parametrize('knn_metric', ['minkowski', 'mahalanobis'])
def test_valid_knn_metric(knn_metric, create_X_y):
    X, y = create_X_y
    ds = BaseDS(knn_metric=knn_metric)
    ds.fit(X, y)


@pytest.mark.parametrize('knn_metric', ['invalidValue'])
def test_invalid_knn_metric(knn_metric, create_X_y):
    X, y = create_X_y
    with pytest.raises(ValueError):
        ds = BaseDS(knn_metric=knn_metric)
        ds.fit(X, y)


def test_minkowski_metric():
    X, y = make_classification()
    ds_test = BaseDS(knn_metric='minkowski')
    ds_test._set_region_of_competence_algorithm(X)
    assert ds_test.roc_algorithm_.metric == 'minkowski'
    assert ds_test.roc_algorithm_.metric_params is None


def test_mahalanobis_metric():
    X, y = make_classification()
    ds_test = BaseDS(knn_metric='mahalanobis')
    ds_test._set_region_of_competence_algorithm(X)
    assert ds_test.roc_algorithm_.metric == 'mahalanobis'
    assert np.array_equal(ds_test.roc_algorithm_.metric_params['VI'], np.cov(X))


# ----- Integration tests -----

def setup_classifiers():
    rng = np.random.RandomState(654321)

    # Generate a classification dataset
    X, y = make_classification(n_classes=2, n_samples=1000, weights=[0.2, 0.8],
                               random_state=rng)
    # split the data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=rng)

    # Scale the variables to have 0 mean and unit variance
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)

    # Split the data into training and DSEL for DS techniques
    X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train,
                                                        test_size=0.5,
                                                        random_state=rng)
    # Considering a pool composed of 10 base classifiers
    pool_classifiers = RandomForestClassifier(n_estimators=10, n_jobs=-1,
                                              random_state=rng, max_depth=10)
    pool_classifiers.fit(X_train, y_train)
    return pool_classifiers, X_dsel, y_dsel, X_test, y_test


@pytest.mark.parametrize('metric', [
    ['minkowski', 0.9],
    ['mahalanobis', 0.8909090909090909]
])
def test_knorau(metric):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    technique = KNORAU(pool_classifiers, knn_metric=metric[0])
    technique.fit(X_dsel, y_dsel)
    assert np.isclose(technique.score(X_test, y_test), metric[1])


@pytest.mark.parametrize('metric', [
    ['minkowski', 0.9],
    ['mahalanobis', 0.8818181818181818]
])
def test_knorae(metric):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    technique = KNORAE(pool_classifiers, knn_metric=metric[0])
    technique.fit(X_dsel, y_dsel)
    assert np.isclose(technique.score(X_test, y_test), metric[1])


@pytest.mark.parametrize('metric', [
    ['minkowski', 0.8939393939393939],
    ['mahalanobis', 0.8909090909090909]
])
def test_desp(metric):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    technique = DESP(pool_classifiers, knn_metric=metric[0])
    technique.fit(X_dsel, y_dsel)
    assert np.isclose(technique.score(X_test, y_test), metric[1])


@pytest.mark.parametrize('metric', [
    ['minkowski', 0.8818181818181818],
    ['mahalanobis', 0.8727272727272727]
])
def test_ola(metric):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    technique = OLA(pool_classifiers, knn_metric=metric[0])
    technique.fit(X_dsel, y_dsel)
    assert np.isclose(technique.score(X_test, y_test), metric[1])


@pytest.mark.parametrize('metric', [
    ['minkowski', 0.8666666666666667],
    ['mahalanobis', 0.8606060606060606]
])
def test_mcb(metric):
    rng = np.random.RandomState(123456)
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    technique = MCB(pool_classifiers, random_state=rng, knn_metric=metric[0])
    technique.fit(X_dsel, y_dsel)
    assert np.isclose(technique.score(X_test, y_test), metric[1])


@pytest.mark.parametrize('metric', [
    ['minkowski', 0.8787878787878788],
    ['mahalanobis', 0.8757575757575757]
])
def test_rank(metric):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    technique = Rank(pool_classifiers, knn_metric=metric[0])
    technique.fit(X_dsel, y_dsel)
    assert np.isclose(technique.score(X_test, y_test), metric[1])


@pytest.mark.parametrize('metric', [
    ['minkowski', 0.9121212121212121],
    ['mahalanobis', 0.8515151515151516]
])
def test_aposteriori(metric):
    rng = np.random.RandomState(123456)
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    technique = APosteriori(
        pool_classifiers, random_state=rng, knn_metric=metric[0])
    technique.fit(X_dsel, y_dsel)
    assert np.isclose(technique.score(X_test, y_test), metric[1])


@pytest.mark.parametrize('metric', [
    ['minkowski', 0.9],
    ['mahalanobis', 0.906060606060606]
])
def test_meta(metric):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    technique = METADES(pool_classifiers, knn_metric=metric[0])
    technique.fit(X_dsel, y_dsel)
    assert np.isclose(technique.score(X_test, y_test), metric[1])


@pytest.mark.parametrize('metric', [
    ['minkowski', 0.896969696969697],
    ['mahalanobis', 0.8939393939393939]
])
def test_desknn(metric):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    technique = DESKNN(pool_classifiers, knn_metric=metric[0])
    technique.fit(X_dsel, y_dsel)
    assert np.isclose(technique.score(X_test, y_test), metric[1])


@pytest.mark.parametrize('metric', [
    ['minkowski', 0.9030303030303031],
    ['mahalanobis', 0.8939393939393939]
])
def test_deskl(metric):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    technique = DESKL(pool_classifiers, knn_metric=metric[0])
    technique.fit(X_dsel, y_dsel)
    assert np.isclose(technique.score(X_test, y_test), metric[1])
