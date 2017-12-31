import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from pythonds.dcs.a_priori import APriori
from pythonds.dcs.mcb import MCB
# Example of a dcs techniques
from pythonds.dcs.ola import OLA
from pythonds.des.des_p import DESP
# Example of a des techniques
from pythonds.des.knora_e import KNORAE
from pythonds.des.knora_u import KNORAU


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


def test_baseline():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    # Calculate classification accuracy of each technique
    assert np.isclose(pool_classifiers.score(X_test, y_test), 0.97872340425531912)


