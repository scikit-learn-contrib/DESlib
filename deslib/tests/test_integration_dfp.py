import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# DCS techniques
from deslib.dcs.a_posteriori import APosteriori
from deslib.dcs.a_priori import APriori
from deslib.dcs.lca import LCA
from deslib.dcs.mcb import MCB
from deslib.dcs.mla import MLA
from deslib.dcs.ola import OLA
from deslib.dcs.rank import Rank
# DES techniques
from deslib.des.des_knn import DESKNN
from deslib.des.des_p import DESP
from deslib.des.knop import KNOP
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU
from deslib.des.meta_des import METADES
from deslib.des import DESKL
# Static techniques
from deslib.static.oracle import Oracle
from deslib.static.single_best import SingleBest
from deslib.static.static_selection import StaticSelection


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
    pool_classifiers = RandomForestClassifier(n_estimators=10,
                                              random_state=rng, max_depth=10)
    pool_classifiers.fit(X_train, y_train)
    return pool_classifiers, X_dsel, y_dsel, X_test, y_test


def test_knorau():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    knorau = KNORAU(pool_classifiers, DFP=True)
    knorau.fit(X_dsel, y_dsel)
    assert np.isclose(knorau.score(X_test, y_test), 0.90606060606060601)


def test_kne():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    kne = KNORAE(pool_classifiers, DFP=True)
    kne.fit(X_dsel, y_dsel)
    assert np.isclose(kne.score(X_test, y_test), 0.896969696969697)


def test_desp():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    desp = DESP(pool_classifiers, DFP=True)
    desp.fit(X_dsel, y_dsel)
    assert np.isclose(desp.score(X_test, y_test), 0.8939393939393939)


def test_ola():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    ola = OLA(pool_classifiers, DFP=True)
    ola.fit(X_dsel, y_dsel)
    assert np.isclose(ola.score(X_test, y_test), 0.88181818181818183)


def test_lca():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    lca = LCA(pool_classifiers, DFP=True)
    lca.fit(X_dsel, y_dsel)
    assert np.isclose(lca.score(X_test, y_test), 0.88787878787878793)


def test_MLA():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    mla = MLA(pool_classifiers, DFP=True)
    mla.fit(X_dsel, y_dsel)
    assert np.isclose(mla.score(X_test, y_test), 0.88787878787878793)


def test_mcb():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)

    mcb = MCB(pool_classifiers, random_state=rng, DFP=True)
    mcb.fit(X_dsel, y_dsel)
    assert np.isclose(mcb.score(X_test, y_test), 0.8606060606060606)


def test_apriori():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)

    apriori = APriori(pool_classifiers, random_state=rng, DFP=True)
    apriori.fit(X_dsel, y_dsel)
    assert np.isclose(apriori.score(X_test, y_test), 0.87272727272727268)


def test_rank():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    rank = Rank(pool_classifiers, DFP=True)
    rank.fit(X_dsel, y_dsel)
    assert np.isclose(rank.score(X_test, y_test), 0.8787878787878788)


def test_aposteriori():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)

    a_posteriori = APosteriori(pool_classifiers, random_state=rng, DFP=True)
    a_posteriori.fit(X_dsel, y_dsel)
    assert np.isclose(a_posteriori.score(X_test, y_test), 0.90000000000000002)


# for some reason, the result of the META-DES is different if all tests are
# executed together than if this test script is executed alone. So, here we are
# accepting both values.
def test_meta():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    meta_des = METADES(pool_classifiers, DFP=True)
    meta_des.fit(X_dsel, y_dsel)
    assert np.isclose(meta_des.score(X_test, y_test), 0.9151515151515152)


def test_knop():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    kne = KNOP(pool_classifiers, DFP=True)
    kne.fit(X_dsel, y_dsel)

    assert np.isclose(kne.score(X_test, y_test), 0.9030303030303031)


def test_desknn():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    desknn = DESKNN(pool_classifiers, DFP=True)
    desknn.fit(X_dsel, y_dsel)
    assert np.isclose(desknn.score(X_test, y_test), 0.89393939393939392)


def test_deskl():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    deskl = DESKL(pool_classifiers, DFP=True)
    deskl.fit(X_dsel, y_dsel)
    assert np.isclose(deskl.score(X_test, y_test), 0.90303030303030307)


def test_oracle():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    oracle = Oracle(pool_classifiers)
    oracle.fit(X_dsel, y_dsel)
    assert np.isclose(oracle.score(X_test, y_test), 0.98787878787878791)


def test_single_best():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    single_best = SingleBest(pool_classifiers)
    single_best.fit(X_dsel, y_dsel)
    assert np.isclose(single_best.score(X_test, y_test), 0.86969696969696975)


def test_static_selection():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    static_selection = StaticSelection(pool_classifiers)
    static_selection.fit(X_dsel, y_dsel)
    assert np.isclose(static_selection.score(X_test, y_test),
                      0.90606060606060601)


# --------------------- Testing predict_proba ---------------------------------
def test_kne_proba():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    kne = KNORAE(pool_classifiers, DFP=True)
    kne.fit(X_dsel, y_dsel)
    probas = kne.predict_proba(X_test)
    expected = np.load('deslib/tests/expected_values/kne_proba_DFP.npy')
    assert np.allclose(probas, expected)


def test_desp_proba():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    desp = DESP(pool_classifiers, DFP=True)
    desp.fit(X_dsel, y_dsel)
    probas = desp.predict_proba(X_test)
    expected = np.load('deslib/tests/expected_values/desp_proba_DFP.npy')
    assert np.allclose(probas, expected)


def test_ola_proba():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    ola = OLA(pool_classifiers, DFP=True)
    ola.fit(X_dsel, y_dsel)
    probas = ola.predict_proba(X_test)
    expected = np.load('deslib/tests/expected_values/ola_proba_DFP.npy')
    assert np.allclose(probas, expected)


def test_mcb_proba():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)

    mcb = MCB(pool_classifiers, random_state=rng, DFP=True)
    mcb.fit(X_dsel, y_dsel)
    probas = mcb.predict_proba(X_test)
    expected = np.load('deslib/tests/expected_values/mcb_proba_DFP.npy')
    assert np.allclose(probas, expected)


def test_desknn_proba():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    desknn = DESKNN(pool_classifiers, DFP=True)
    desknn.fit(X_dsel, y_dsel)
    probas = desknn.predict_proba(X_test)
    expected = np.load('deslib/tests/expected_values/desknn_probas_DFP.npy')
    assert np.allclose(probas, expected)
