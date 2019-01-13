import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import make_classification
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# DCS techniques
from deslib.dcs.a_posteriori import APosteriori
from deslib.dcs.mcb import MCB
from deslib.dcs.ola import OLA
# DES techniques
from deslib.des.des_p import DESP
from deslib.des.knora_u import KNORAU
from deslib.des import DESClustering


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
    model = CalibratedClassifierCV(Perceptron(max_iter=5))

    pool_classifiers = BaggingClassifier(model, n_estimators=100,
                                         random_state=rng)
    pool_classifiers.fit(X_train, y_train)
    return pool_classifiers, X_dsel, y_dsel, X_test, y_test


def test_knorau():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    knorau = KNORAU(pool_classifiers, DFP=True, with_IH=True, IH_rate=0.1)
    knorau.fit(X_dsel, y_dsel)
    assert np.isclose(knorau.score(X_test, y_test), 0.9090909090909091)


def test_desp():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    desp = DESP(pool_classifiers, DFP=True, with_IH=True, IH_rate=0.1)
    desp.fit(X_dsel, y_dsel)
    assert np.isclose(desp.score(X_test, y_test), 0.9090909090909091)


def test_ola():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    ola = OLA(pool_classifiers, DFP=True, with_IH=True, IH_rate=0.1)
    ola.fit(X_dsel, y_dsel)
    assert np.isclose(ola.score(X_test, y_test), 0.88181818181818183)


def test_mcb():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)

    mcb = MCB(pool_classifiers, random_state=rng, DFP=True, with_IH=True,
              IH_rate=0.1)
    mcb.fit(X_dsel, y_dsel)
    assert np.isclose(mcb.score(X_test, y_test), 0.9)


def test_aposteriori():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)

    a_posteriori = APosteriori(pool_classifiers, random_state=rng, DFP=True,
                               with_IH=True, IH_rate=0.1)
    a_posteriori.fit(X_dsel, y_dsel)
    assert np.isclose(a_posteriori.score(X_test, y_test), 0.8303030303030303)


def test_des_clustering():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)

    des_clustering = DESClustering(pool_classifiers, random_state=rng,
                                   with_IH=True, IH_rate=0.28)
    des_clustering.fit(X_dsel, y_dsel)
    assert np.isclose(des_clustering.score(X_test, y_test), 0.906060606060606)
