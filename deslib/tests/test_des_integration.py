import itertools
import warnings

import numpy as np
import pytest
from sklearn.cluster import KMeans
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# DCS techniques
from deslib.dcs import APosteriori
from deslib.dcs import APriori
from deslib.dcs import LCA
from deslib.dcs import MCB
from deslib.dcs import MLA
from deslib.dcs import OLA
from deslib.dcs import Rank
# DES techniques
from deslib.des import DESClustering
from deslib.des import DESKNN
from deslib.des import DESP
from deslib.des import KNOP
from deslib.des import KNORAE
from deslib.des import KNORAU
from deslib.des import METADES
from deslib.des import RRC, MinimumDifference, DESKL
# Static techniques
from deslib.static import Oracle
from deslib.static import SingleBest
from deslib.static import StackedClassifier
from deslib.static import StaticSelection
from deslib.util import faiss_knn_wrapper


@pytest.mark.skip(
    reason='Need to wait for changes on scikit-learn (see issue #89)')
def test_grid_search():
    # This tests if the estimator can be cloned and used in a grid search
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    kne = KNORAE(pool_classifiers)
    params = {'k': [1, 3, 5, 7]}
    grid = GridSearchCV(kne, params)
    grid.fit(X_dsel, y_dsel)
    grid.best_estimator_.score(X_test, y_test)


knn_methods = [None]
voting = ['hard', 'soft']

if faiss_knn_wrapper.is_available():
    # knn_methods.append(faiss_knn_wrapper.FaissKNNClassifier)
    knn_methods.append('faiss')
else:
    warnings.warn("Not testing FAISS for KNN")


def test_label_encoder_integration_list_classifiers():
    rng = np.random.RandomState(123456)
    X_dsel, X_test, X_train, y_dsel, y_test, y_train = load_dataset(
        encode_labels=['no', 'yes'], rng=rng)

    pool_classifiers = [LogisticRegression(), SVC(probability=True)]
    [clf.fit(X_train, y_train) for clf in pool_classifiers]

    knorau = KNORAU(pool_classifiers)
    knorau.fit(X_dsel, y_dsel)

    this_score = knorau.score(X_test, y_test)
    assert np.isclose(this_score, 0.9787234042553191)


def test_label_encoder_integration_sklearn_ensembles():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers(
        encode_labels=['no', 'yes'])

    knorau = KNORAU(pool_classifiers)
    knorau.fit(X_dsel, y_dsel)
    assert np.isclose(knorau.score(X_test, y_test), 0.9787234042553191)


def test_label_encoder_integration_sklearn_ensembles_not_encoding():

    rng = np.random.RandomState(123456)
    X_dsel, X_test, X_train, y_dsel, y_test, y_train = load_dataset(
        ['yes', 'no'], rng)

    # Train a pool of using adaboost which has label encoding problems.
    pool_classifiers = AdaBoostClassifier(n_estimators=10, random_state=rng)
    pool_classifiers.fit(X_train, y_train)

    knorau = KNORAU(pool_classifiers)
    knorau.fit(X_dsel, y_dsel)
    assert np.isclose(knorau.score(X_test, y_test), 0.9627659574468085)


def setup_classifiers(encode_labels=None):
    rng = np.random.RandomState(123456)

    X_dsel, X_test, X_train, y_dsel, y_test, y_train = load_dataset(
        encode_labels, rng)
    model = LogisticRegression(C=1, random_state=rng)
    # Train a pool of 100 classifiers
    pool_classifiers = BaggingClassifier(model, n_estimators=100, n_jobs=-1,
                                         random_state=rng)
    pool_classifiers.fit(X_train, y_train)
    return pool_classifiers, X_dsel, y_dsel, X_test, y_test


def load_dataset(encode_labels, rng):
    # Generate a classification dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    if encode_labels is not None:
        y = np.take(encode_labels, y)
    # split the data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=rng)
    # Scale the variables to have 0 mean and unit variance
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    # Split the data into training and DSEL for DS techniques
    X_dsel, y_dsel = X_train, y_train
    # Considering a pool composed of 10 base classifiers
    # Calibrating Perceptrons to estimate probabilities
    return X_dsel, X_test, X_train, y_dsel, y_test, y_train


@pytest.mark.parametrize('knn_methods', knn_methods)
def test_knorau(knn_methods):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    knorau = KNORAU(pool_classifiers, knn_classifier=knn_methods)
    knorau.fit(X_dsel, y_dsel)
    assert np.isclose(knorau.score(X_test, y_test), 0.9787234042553191)


@pytest.mark.parametrize('knn_methods, voting',
                         itertools.product(knn_methods, voting))
def test_kne(knn_methods, voting):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    kne = KNORAE(pool_classifiers, knn_classifier=knn_methods, voting=voting)
    kne.fit(X_dsel, y_dsel)
    assert np.isclose(kne.score(X_test, y_test), 0.9787234042553191)


@pytest.mark.parametrize('knn_methods, voting',
                         itertools.product(knn_methods, voting))
def test_desp(knn_methods, voting):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    desp = DESP(pool_classifiers, knn_classifier=knn_methods, voting=voting)
    desp.fit(X_dsel, y_dsel)
    assert np.isclose(desp.score(X_test, y_test), 0.9787234042553191)


@pytest.mark.parametrize('knn_methods', knn_methods)
def test_ola(knn_methods):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    ola = OLA(pool_classifiers, knn_classifier=knn_methods)
    ola.fit(X_dsel, y_dsel)
    assert np.isclose(ola.score(X_test, y_test), 0.9787234042553191)


@pytest.mark.parametrize('knn_methods', knn_methods)
def test_lca(knn_methods):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    lca = LCA(pool_classifiers, knn_classifier=knn_methods)
    lca.fit(X_dsel, y_dsel)
    assert np.isclose(lca.score(X_test, y_test), 0.973404255319149)


@pytest.mark.parametrize('knn_methods', knn_methods)
def test_MLA(knn_methods):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    mla = MLA(pool_classifiers, knn_classifier=knn_methods)
    mla.fit(X_dsel, y_dsel)
    assert np.isclose(mla.score(X_test, y_test), 0.973404255319149)


@pytest.mark.parametrize('knn_methods', knn_methods)
def test_mcb(knn_methods):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)

    mcb = MCB(pool_classifiers, random_state=rng, knn_classifier=knn_methods)
    mcb.fit(X_dsel, y_dsel)
    assert np.isclose(mcb.score(X_test, y_test), 0.9627659574468085)


def test_apriori():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)

    apriori = APriori(pool_classifiers, random_state=rng)
    apriori.fit(X_dsel, y_dsel)
    assert np.isclose(apriori.score(X_test, y_test), 0.9680851063829787)


@pytest.mark.parametrize('knn_methods', knn_methods)
def test_rank(knn_methods):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    rank = Rank(pool_classifiers, knn_classifier=knn_methods)
    rank.fit(X_dsel, y_dsel)
    assert np.isclose(rank.score(X_test, y_test), 0.9787234042553191)


def test_aposteriori():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)

    a_posteriori = APosteriori(pool_classifiers, random_state=rng)
    a_posteriori.fit(X_dsel, y_dsel)
    assert np.isclose(a_posteriori.score(X_test, y_test), 0.9787234042553191)


@pytest.mark.parametrize('knn_methods, voting',
                         itertools.product(knn_methods, voting))
def test_meta(knn_methods, voting):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    meta_des = METADES(pool_classifiers,
                       knn_classifier=knn_methods, voting=voting)
    meta_des.fit(X_dsel, y_dsel)
    assert np.isclose(meta_des.score(X_test, y_test), 0.9787234042553191)


@pytest.mark.parametrize('voting', voting)
def test_rrc(voting):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    rrc = RRC(pool_classifiers, voting=voting)
    rrc.fit(X_dsel, y_dsel)
    assert np.isclose(rrc.score(X_test, y_test), 0.9840425531914894)


@pytest.mark.parametrize('knn_methods', knn_methods)
def test_deskl(knn_methods):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    deskl = DESKL(pool_classifiers, knn_classifier=knn_methods)
    deskl.fit(X_dsel, y_dsel)
    assert np.isclose(deskl.score(X_test, y_test), 0.9787234042553191)


@pytest.mark.parametrize('knn_methods', knn_methods)
def test_minimum_diff(knn_methods):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    minimum_diff = MinimumDifference(pool_classifiers,
                                     knn_classifier=knn_methods)
    minimum_diff.fit(X_dsel, y_dsel)
    assert np.isclose(minimum_diff.score(X_test, y_test), 0.9787234042553191)


@pytest.mark.parametrize('knn_methods, voting',
                         itertools.product(knn_methods, voting))
def test_knop(knn_methods, voting):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    knop = KNOP(pool_classifiers, knn_classifier=knn_methods, voting=voting)
    knop.fit(X_dsel, y_dsel)
    assert np.isclose(knop.score(X_test, y_test), 0.9787234042553191)


@pytest.mark.parametrize('knn_methods, voting',
                         itertools.product(knn_methods, voting))
def test_desknn(knn_methods, voting):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    desknn = DESKNN(pool_classifiers,
                    knn_classifier=knn_methods, voting=voting)
    desknn.fit(X_dsel, y_dsel)
    assert np.isclose(desknn.score(X_test, y_test), 0.9787234042553191)


@pytest.mark.parametrize('voting', voting)
def test_des_clustering(voting):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)
    des_clustering = DESClustering(pool_classifiers,
                                   random_state=rng, voting=voting)
    des_clustering.fit(X_dsel, y_dsel)
    assert np.isclose(des_clustering.score(X_test, y_test),
                      0.973404255319149)


def test_oracle():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    oracle = Oracle(pool_classifiers)
    oracle.fit(X_dsel, y_dsel)
    assert np.isclose(oracle.score(X_test, y_test), 0.99468085106382975)


def test_single_best():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    single_best = SingleBest(pool_classifiers)
    single_best.fit(X_dsel, y_dsel)
    assert np.isclose(single_best.score(X_test, y_test), 0.973404255319149)


def test_static_selection():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    static_selection = StaticSelection(pool_classifiers)
    static_selection.fit(X_dsel, y_dsel)
    assert np.isclose(static_selection.score(X_test, y_test),
                      0.9787234042553191)


# ------------------------ Testing predict_proba ------------------------------
@pytest.mark.parametrize('knn_methods', knn_methods)
def test_kne_proba(knn_methods):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    kne = KNORAE(pool_classifiers, knn_classifier=knn_methods, voting='soft')
    kne.fit(X_dsel, y_dsel)
    probas = kne.predict_proba(X_test)
    expected = np.load(
        'deslib/tests/expected_values/kne_proba_integration.npy')
    assert np.allclose(probas, expected, atol=0.01)


@pytest.mark.parametrize('knn_methods', knn_methods)
def test_desp_proba(knn_methods):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    desp = DESP(pool_classifiers, knn_classifier=knn_methods, voting='soft')
    desp.fit(X_dsel, y_dsel)
    probas = desp.predict_proba(X_test)
    expected = np.load(
        'deslib/tests/expected_values/desp_proba_integration.npy')
    assert np.allclose(probas, expected, atol=0.01)


@pytest.mark.parametrize('knn_methods', knn_methods)
def test_ola_proba(knn_methods):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    ola = OLA(pool_classifiers, knn_classifier=knn_methods)
    ola.fit(X_dsel, y_dsel)
    probas = ola.predict_proba(X_test)
    expected = np.load(
        'deslib/tests/expected_values/ola_proba_integration.npy')
    assert np.allclose(probas, expected, atol=0.01)


@pytest.mark.parametrize('knn_methods', knn_methods)
def test_mcb_proba(knn_methods):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)

    mcb = MCB(pool_classifiers, random_state=rng, knn_classifier=knn_methods)

    mcb.fit(X_dsel, y_dsel)
    probas = mcb.predict_proba(X_test).argmax(axis=1)
    expected = mcb.predict(X_test)
    assert np.allclose(probas, expected)


@pytest.mark.parametrize('knn_methods', knn_methods)
def test_desknn_proba(knn_methods):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    desknn = DESKNN(pool_classifiers, knn_classifier=knn_methods,
                    voting='soft')
    desknn.fit(X_dsel, y_dsel)
    probas = desknn.predict_proba(X_test).argmax(axis=1)
    expected = desknn.predict(X_test)
    assert np.allclose(probas, expected)


def test_des_clustering_proba():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)
    cluster = KMeans(n_clusters=5, random_state=rng)
    des_clustering = DESClustering(pool_classifiers, clustering=cluster,
                                   voting='soft')
    des_clustering.fit(X_dsel, y_dsel)
    probas = des_clustering.predict_proba(X_test).argmax(axis=1)
    expected = des_clustering.predict(X_test)
    assert np.allclose(probas, expected)


@pytest.mark.parametrize('knn_methods', knn_methods)
def test_knop_proba(knn_methods):
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    knop = KNOP(pool_classifiers, knn_classifier=knn_methods, voting='soft')
    knop.fit(X_dsel, y_dsel)
    probas = knop.predict_proba(X_test)
    expected = np.load(
        'deslib/tests/expected_values/knop_proba_integration.npy')
    assert np.allclose(probas, expected, atol=0.01)


def test_meta_no_pool_of_classifiers():
    rng = np.random.RandomState(123456)
    data = load_breast_cancer()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                        random_state=rng)
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)
    meta_des = METADES(random_state=rng, DSEL_perc=0.5)
    meta_des.fit(X_train, y_train)
    assert np.isclose(meta_des.score(X_test, y_test), 0.898936170212766)


def test_ola_subspaces():
    rng = np.random.RandomState(123456)
    X_dsel, X_test, X_train, y_dsel, y_test, y_train = load_dataset(
        None, rng)
    pool = BaggingClassifier(LogisticRegression(),
                             bootstrap_features=True,
                             max_features=0.5,
                             random_state=rng).fit(X_train, y_train)

    ola = OLA(pool)
    ola.fit(X_dsel, y_dsel)
    assert np.isclose(ola.score(X_test, y_test),
                      0.9680851063829787)


def test_knorae_subspaces():
    rng = np.random.RandomState(123456)
    X_dsel, X_test, X_train, y_dsel, y_test, y_train = load_dataset(
        None, rng)
    # split the data into training and test data
    pool = BaggingClassifier(LogisticRegression(),
                             max_features=0.5,
                             random_state=rng).fit(X_train, y_train)

    knorae = KNORAE(pool)
    knorae.fit(X_dsel, y_dsel)
    assert np.isclose(knorae.score(X_test, y_test),
                      0.9787234042553191)


def test_knorae_subspaces():
    rng = np.random.RandomState(123456)
    X_dsel, X_test, X_train, y_dsel, y_test, y_train = load_dataset(
        None, rng)
    pool = BaggingClassifier(LogisticRegression(),
                             max_features=0.5,
                             random_state=rng).fit(X_train, y_train)

    knorae = KNORAE(pool)
    knorae.fit(X_dsel, y_dsel)
    y_pred = knorae.predict_proba(X_test).argmax(axis=1)
    assert np.isclose(accuracy_score(y_pred, y_test),
                      0.9787234042553191)


def test_oracle_subspaces():
    rng = np.random.RandomState(123456)
    X_dsel, X_test, X_train, y_dsel, y_test, y_train = load_dataset(
        None, rng)
    pool = BaggingClassifier(LogisticRegression(),
                             max_features=0.5,
                             random_state=rng).fit(X_train, y_train)

    oracle = Oracle(pool)
    oracle.fit(X_dsel, y_dsel)
    assert np.isclose(oracle.score(X_test, y_test),
                      0.9946808510638298)


def test_oracle_subspaces_proba():
    rng = np.random.RandomState(123456)
    X_dsel, X_test, X_train, y_dsel, y_test, y_train = load_dataset(
        None, rng)
    pool = BaggingClassifier(LogisticRegression(),
                             max_features=0.5,
                             random_state=rng).fit(X_train, y_train)

    oracle = Oracle(pool)
    oracle.fit(X_dsel, y_dsel)
    y_pred = oracle.predict_proba(X_test, y_test).argmax(axis=1)
    assert np.isclose(accuracy_score(y_pred, y_test),
                      0.9946808510638298)


def test_static_selection_subspaces():
    rng = np.random.RandomState(123456)
    X_dsel, X_test, X_train, y_dsel, y_test, y_train = load_dataset(
        None, rng)
    pool = BaggingClassifier(LogisticRegression(),
                             max_features=0.5,
                             random_state=rng).fit(X_train, y_train)

    static = StaticSelection(pool)
    static.fit(X_dsel, y_dsel)
    assert np.isclose(static.score(X_test, y_test),
                      0.9787234042553191)


def test_static_selection_subspaces_proba():
    rng = np.random.RandomState(123456)
    X_dsel, X_test, X_train, y_dsel, y_test, y_train = load_dataset(
        None, rng)
    pool = BaggingClassifier(LogisticRegression(),
                             max_features=0.5,
                             random_state=rng).fit(X_train, y_train)

    static = StaticSelection(pool)
    static.fit(X_dsel, y_dsel)
    y_pred_proba = static.predict_proba(X_test).argmax(axis=1)
    y_pred = static.predict(X_test)
    assert np.isclose(y_pred, y_pred_proba).all()


def test_stacked_subspaces():
    rng = np.random.RandomState(123456)
    X_dsel, X_test, X_train, y_dsel, y_test, y_train = load_dataset(
        None, rng)
    pool = BaggingClassifier(LogisticRegression(),
                             max_features=0.5,
                             random_state=rng).fit(X_train, y_train)

    stacked = StackedClassifier(pool)
    stacked.fit(X_dsel, y_dsel)
    assert np.isclose(stacked.score(X_test, y_test),
                      0.973404255319149)


def test_stacked_subspaces_proba():
    rng = np.random.RandomState(123456)
    X_dsel, X_test, X_train, y_dsel, y_test, y_train = load_dataset(
        None, rng)
    pool = BaggingClassifier(LogisticRegression(),
                             max_features=0.5,
                             random_state=rng).fit(X_train, y_train)

    stacked = StackedClassifier(pool)
    stacked.fit(X_dsel, y_dsel)
    y_pred = stacked.predict_proba(X_test).argmax(axis=1)
    assert np.isclose(accuracy_score(y_pred, y_test),
                      0.973404255319149)


def test_single_best_subspaces():
    rng = np.random.RandomState(123456)
    X_dsel, X_test, X_train, y_dsel, y_test, y_train = load_dataset(
        None, rng)
    pool = BaggingClassifier(LogisticRegression(),
                             max_features=0.5,
                             random_state=rng).fit(X_train, y_train)

    single_best = SingleBest(pool)
    single_best.fit(X_dsel, y_dsel)
    assert np.isclose(single_best.score(X_test, y_test),
                      0.9627659574468085)


def test_single_best_subspaces_proba():
    rng = np.random.RandomState(123456)
    X_dsel, X_test, X_train, y_dsel, y_test, y_train = load_dataset(
        None, rng)
    pool = BaggingClassifier(LogisticRegression(),
                             max_features=0.5,
                             random_state=rng).fit(X_train, y_train)

    single_best = SingleBest(pool)
    single_best.fit(X_dsel, y_dsel)
    y_pred = single_best.predict_proba(X_test).argmax(axis=1)

    assert np.isclose(accuracy_score(y_pred, y_test),
                      0.9627659574468085)
