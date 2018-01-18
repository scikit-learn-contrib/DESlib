import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from deslib.dcs.a_posteriori import APosteriori
# DCS techniques
from deslib.dcs.a_priori import APriori
from deslib.dcs.lca import LCA
from deslib.dcs.mcb import MCB
from deslib.dcs.mla import MLA
from deslib.dcs.ola import OLA
from deslib.dcs.rank import Rank
# DES techniques
from deslib.des.des_clustering import DESClustering
from deslib.des.des_knn import DESKNN
from deslib.des.des_p import DESP
from deslib.des.knop import KNOP
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU
from deslib.des.meta_des import METADES
from deslib.des.probabilistic import RRC, MinimumDifference, DESKL
# Static techniques
from deslib.static.oracle import Oracle
from deslib.static.single_best import SingleBest
from deslib.static.static_selection import StaticSelection


def setup_classifiers():
    rng = np.random.RandomState(123456)

    # Generate a classification dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    # split the data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=rng)

    # Scale the variables to have 0 mean and unit variance
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)

    # Split the data into training and DSEL for DS techniques
    X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=0.5, random_state=rng)
    # Considering a pool composed of 10 base classifiers
    # Calibrating Perceptrons to estimate probabilities
    model = CalibratedClassifierCV(Perceptron(max_iter=5))
    # Train a pool of 100 classifiers
    pool_classifiers = BaggingClassifier(model, n_estimators=10, random_state=rng)
    pool_classifiers.fit(X_train, y_train)
    return pool_classifiers, X_dsel, y_dsel, X_test, y_test


def test_knorau():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    knorau = KNORAU(pool_classifiers)
    knorau.fit(X_dsel, y_dsel)
    assert np.isclose(knorau.score(X_test, y_test), 0.97340425531914898)


def test_kne():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    kne = KNORAE(pool_classifiers)
    kne.fit(X_dsel, y_dsel)
    assert np.isclose(kne.score(X_test, y_test), 0.973404255319148)


def test_desp():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    desp = DESP(pool_classifiers)
    desp.fit(X_dsel, y_dsel)
    assert np.isclose(desp.score(X_test, y_test), 0.97340425531914898)


def test_ola():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    ola = OLA(pool_classifiers)
    ola.fit(X_dsel, y_dsel)
    assert np.isclose(ola.score(X_test, y_test), 0.96808510638297873)


def test_lca():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    lca = LCA(pool_classifiers)
    lca.fit(X_dsel, y_dsel)
    assert np.isclose(lca.score(X_test, y_test), 0.96808510638297873)


def test_MLA():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    mla = MLA(pool_classifiers)
    mla.fit(X_dsel, y_dsel)
    assert np.isclose(mla.score(X_test, y_test), 0.96808510638297873)


def test_mcb():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)

    mcb = MCB(pool_classifiers, rng=rng)
    mcb.fit(X_dsel, y_dsel)
    assert np.isclose(mcb.score(X_test, y_test), 0.96276595744680848)


def test_apriori():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)

    apriori = APriori(pool_classifiers, rng=rng)
    apriori.fit(X_dsel, y_dsel)
    assert np.isclose(apriori.score(X_test, y_test), 0.97872340425531912)


def test_rank():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    rank = Rank(pool_classifiers)
    rank.fit(X_dsel, y_dsel)
    assert np.isclose(rank.score(X_test, y_test), 0.96276595744680848)


def test_aposteriori():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)

    a_posteriori = APosteriori(pool_classifiers, rng=rng)
    a_posteriori.fit(X_dsel, y_dsel)
    assert np.isclose(a_posteriori.score(X_test, y_test), 0.96276595744680848)


def test_meta():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    meta_des = METADES(pool_classifiers)
    meta_des.fit(X_dsel, y_dsel)
    assert np.isclose(meta_des.score(X_test, y_test), 0.97872340425531912)


def test_rrc():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    rrc = RRC(pool_classifiers)
    rrc.fit(X_dsel, y_dsel)
    assert np.isclose(rrc.score(X_test, y_test), 0.97340425531914898)


def test_deskl():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    deskl = DESKL(pool_classifiers)
    deskl.fit(X_dsel, y_dsel)
    assert np.isclose(deskl.score(X_test, y_test), 0.97340425531914898)


def test_minimum_diff():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    minimum_diff = MinimumDifference(pool_classifiers)
    minimum_diff.fit(X_dsel, y_dsel)
    assert np.isclose(minimum_diff.score(X_test, y_test), 0.97340425531914898)


def test_knop():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    knop = KNOP(pool_classifiers)
    knop.fit(X_dsel, y_dsel)
    assert np.isclose(knop.score(X_test, y_test), 0.97340425531914898)


def test_desknn():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    desknn = DESKNN(pool_classifiers)
    desknn.fit(X_dsel, y_dsel)
    assert np.isclose(desknn.score(X_test, y_test), 0.97340425531914898)


def test_des_clustering():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)

    des_clustering = DESClustering(pool_classifiers, rng=rng)
    des_clustering.fit(X_dsel, y_dsel)
    assert np.isclose(des_clustering.score(X_test, y_test), 0.97872340425531912)


def test_oracle():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    oracle = Oracle(pool_classifiers)
    assert np.isclose(oracle.score(X_test, y_test), 0.99468085106382975)


def test_single_best():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    single_best = SingleBest(pool_classifiers)
    single_best.fit(X_dsel, y_dsel)
    assert np.isclose(single_best.score(X_test, y_test), 0.97872340425531912)


def test_static_selection():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    static_selection = StaticSelection(pool_classifiers)
    static_selection.fit(X_dsel, y_dsel)
    assert np.isclose(static_selection.score(X_test, y_test), 0.96808510638297873)
