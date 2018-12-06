import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# DCS techniques
from deslib.dcs.a_priori import APriori
from deslib.dcs.mcb import MCB
# DES techniques
from deslib.des.des_mi import DESMI
from deslib.des.des_p import DESP
from deslib.des.knop import KNOP
from deslib.des.meta_des import METADES


def setup_classifiers():
    rng = np.random.RandomState(123456)

    X_dsel, X_test, X_train, y_dsel, y_test, y_train = load_dataset(rng)
    # Train a pool of 100 classifiers
    pool_classifiers = AdaBoostClassifier(random_state=rng)
    pool_classifiers.fit(X_train, y_train)
    return pool_classifiers, X_dsel, y_dsel, X_test, y_test


def load_dataset(rng):
    # Generate a classification dataset
    weights = [0.1, 0.2, 0.7]
    X, y = make_classification(n_classes=3, n_samples=2000, n_informative=3,
                               random_state=rng, weights=weights)

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
    # Calibrating Perceptrons to estimate probabilities
    return X_dsel, X_test, X_train, y_dsel, y_test, y_train


def test_desp():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    desp = DESP(pool_classifiers)
    desp.fit(X_dsel, y_dsel)
    assert np.isclose(desp.score(X_test, y_test), 0.6954545454545454)


def test_mcb():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)

    mcb = MCB(pool_classifiers, random_state=rng)
    mcb.fit(X_dsel, y_dsel)
    assert np.isclose(mcb.score(X_test, y_test), 0.7196969696969697)


def test_apriori():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()
    rng = np.random.RandomState(123456)

    apriori = APriori(pool_classifiers, random_state=rng)
    apriori.fit(X_dsel, y_dsel)
    assert np.isclose(apriori.score(X_test, y_test), 0.6878787878787879)


def test_meta():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    meta_des = METADES(pool_classifiers)
    meta_des.fit(X_dsel, y_dsel)
    assert np.isclose(meta_des.score(X_test, y_test), 0.796969696969697)


def test_knop():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    knop = KNOP(pool_classifiers)
    knop.fit(X_dsel, y_dsel)
    assert np.isclose(knop.score(X_test, y_test), 0.8106060606060606)


def test_mi():
    pool_classifiers, X_dsel, y_dsel, X_test, y_test = setup_classifiers()

    desmi = DESMI(pool_classifiers, alpha=0.9)
    desmi.fit(X_dsel, y_dsel)
    assert np.isclose(desmi.score(X_test, y_test), 0.3500000000)
