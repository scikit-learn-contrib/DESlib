"""
In this example we show how to apply different DCS and DES techniques for a classification dataset.

A very important aspect in dynamic selection is the generation of a pool of classifiers. A common practice in the
dynamic selection literature is to use the Bagging (Bootstrap Aggregating) method to generate a pool containing base
classifiers that are both diverse and informative.

In this example we generate a pool of classifiers using the Bagging technique implemented on the Scikit-learn library.
Then, we compare the results obtained by combining this pool of classifiers using the standard Bagging combination
approach versus the application of dynamic selection technique to select the set of most competent classifiers
"""
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Perceptron
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier

from deslib.dcs.ola import OLA
from deslib.dcs.a_priori import APriori
from deslib.dcs.mcb import MCB
from deslib.des.des_p import DESP
from deslib.des.knora_u import KNORAU
from deslib.des.knora_e import KNORAE
from deslib.des.meta_des import METADES

if __name__ == "__main__":
    # Generate a classification dataset
    data = load_breast_cancer()
    X = data.data
    y = data.target
    # split the data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Scale the variables to have 0 mean and unit variance
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Split the data into training and DSEL for DS techniques
    X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=0.5)
    # Considering a pool composed of 10 base classifiers

    # Calibrating Perceptrons to estimate probabilities
    model = CalibratedClassifierCV(Perceptron(max_iter=100))

    # Train a pool of 10 classifiers
    pool_classifiers = BaggingClassifier(model, n_estimators=100)
    pool_classifiers.fit(X_train, y_train)

    # Initialize the DS techniques
    knorau = KNORAU(pool_classifiers)
    kne = KNORAE(pool_classifiers)
    desp = DESP(pool_classifiers)
    ola = OLA(pool_classifiers)
    mcb = MCB(pool_classifiers)
    apriori = APriori(pool_classifiers)
    meta = METADES(pool_classifiers, Hc=0.8)

    # Fit the des techniques
    knorau.fit(X_dsel, y_dsel)
    kne.fit(X_dsel, y_dsel)
    desp.fit(X_dsel, y_dsel)

    # Fit the dcs techniques
    ola.fit(X_dsel, y_dsel)
    mcb.fit(X_dsel, y_dsel)
    apriori.fit(X_dsel, y_dsel)
    meta.fit(X_dsel, y_dsel)

    # Calculate classification accuracy of each technique
    print('Evaluating DS techniques:')
    print('Classification accuracy KNORA-Union: ', knorau.score(X_test, y_test))
    print('Classification accuracy KNORA-Eliminate: ', kne.score(X_test, y_test))
    print('Classification accuracy DESP: ', desp.score(X_test, y_test))
    print('Classification accuracy OLA: ', ola.score(X_test, y_test))
    print('Classification accuracy A priori: ', apriori.score(X_test, y_test))
    print('Classification accuracy META-DES: ', meta.score(X_test, y_test))





