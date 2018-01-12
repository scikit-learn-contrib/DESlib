import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pythonds.dcs.a_posteriori import APosteriori
# DCS techniques
from pythonds.dcs.a_priori import APriori
from pythonds.dcs.lca import LCA
from pythonds.dcs.mcb import MCB
from pythonds.dcs.mla import MLA
from pythonds.dcs.ola import OLA
from pythonds.dcs.rank import Rank
from pythonds.des.des_knn import DESKNN
from pythonds.des.des_p import DESP
from pythonds.des.knop import KNOP
# DES techniques
from pythonds.des.knora_e import KNORAE
from pythonds.des.knora_u import KNORAU
from pythonds.des.meta_des import METADES


def setup_classifiers():
    rng = np.random.RandomState(159841232)

    # Generate a classification dataset
    X, y = make_classification(n_classes=2, n_samples=1000, weights=[0.2, 0.8], random_state=rng)
    # split the data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=rng)

    # Scale the variables to have 0 mean and unit variance
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)

    # Split the data into training and DSEL for DS techniques
    X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=0.5, random_state=rng)
    # Considering a pool composed of 10 base classifiers
    pool_classifiers = RandomForestClassifier(n_estimators=10, random_state=rng)
    pool_classifiers.fit(X_train, y_train)
    return pool_classifiers, X_dsel, y_dsel, X_test, y_test


def test_knorau():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    knorau = KNORAU(pool_classifiers, DFP=True)
    knorau.fit(X_dsel, y_dsel)
    assert np.isclose(knorau.score(X_test, y_test), 0.87272727272727268)


def test_kne():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    kne = KNORAE(pool_classifiers, DFP=True)
    kne.fit(X_dsel, y_dsel)
    assert np.isclose(kne.score(X_test, y_test), 0.84848484848484851)


def test_desp():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    desp = DESP(pool_classifiers)
    desp.fit(X_dsel, y_dsel)
    assert np.isclose(desp.score(X_test, y_test), 0.87878787878787878)


def test_ola():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    ola = OLA(pool_classifiers, DFP=True)
    ola.fit(X_dsel, y_dsel)
    assert np.isclose(ola.score(X_test, y_test), 0.84848484848484851)


def test_lca():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    lca = LCA(pool_classifiers, DFP=True)
    lca.fit(X_dsel, y_dsel)
    assert np.isclose(lca.score(X_test, y_test), 0.82121212121212117)


def test_MLA():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    mla = MLA(pool_classifiers, DFP=True)
    mla.fit(X_dsel, y_dsel)
    assert np.isclose(mla.score(X_test, y_test), 0.82121212121212117)


def test_mcb():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)

    mcb = MCB(pool_classifiers, rng=rng, DFP=True)
    mcb.fit(X_dsel, y_dsel)
    assert np.isclose(mcb.score(X_test, y_test), 0.84242424242424241)


def test_apriori():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)

    apriori = APriori(pool_classifiers, rng=rng, DFP=True)
    apriori.fit(X_dsel, y_dsel)
    assert np.isclose(apriori.score(X_test, y_test), 0.84242424242424241)


def test_rank():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    rank = Rank(pool_classifiers, DFP=True)
    rank.fit(X_dsel, y_dsel)
    assert np.isclose(rank.score(X_test, y_test), 0.80000000000000004)


def test_aposteriori():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)

    a_posteriori = APosteriori(pool_classifiers, rng=rng, DFP=True)
    a_posteriori.fit(X_dsel, y_dsel)
    assert np.isclose(a_posteriori.score(X_test, y_test), 0.84242424242424241)


def test_meta():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    meta_des = METADES(pool_classifiers, DFP=True)
    meta_des.fit(X_dsel, y_dsel)
    assert np.isclose(meta_des.score(X_test, y_test), 0.88484848484848488)


def test_knop():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    knop = KNOP(pool_classifiers, DFP=True)
    knop.fit(X_dsel, y_dsel)
    assert np.isclose(knop.score(X_test, y_test), 0.86969696969696975)


def test_desknn():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    desknn = DESKNN(pool_classifiers, DFP=True)
    desknn.fit(X_dsel, y_dsel)
    assert np.isclose(desknn.score(X_test, y_test), 0.87272727272727268)


