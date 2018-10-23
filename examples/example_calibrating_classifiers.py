# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause
"""
====================================================================
Calibrating base classifiers to estimate probabilities
====================================================================

In this example we show how to apply different DCS and DES techniques for a
classification dataset.

A very important aspect in dynamic selection is the generation of a pool
of classifiers. A common practice in the dynamic selection literature is to
use the Bagging (Bootstrap Aggregating) method to generate a pool containing
base classifiers that are both diverse and informative.

In this example we generate a pool of classifiers using the Bagging technique
implemented on the Scikit-learn library. Then, we compare the results obtained
by combining this pool of classifiers using the standard Bagging combination
approach versus the application of dynamic selection technique to select the
set of most competent classifiers
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from deslib.dcs.a_priori import APriori
from deslib.dcs.mcb import MCB
from deslib.dcs.ola import OLA
from deslib.des.des_p import DESP
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU
from deslib.des.meta_des import METADES

rng = np.random.RandomState(123)

# Generate a classification dataset
data = load_breast_cancer()
X = data.data
y = data.target
# split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state=rng)

# Scale the variables to have 0 mean and unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Split the data into training and DSEL for DS techniques
X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train,
                                                    test_size=0.5,
                                                    random_state=rng)
# Considering a pool composed of 10 base classifiers

# Train a pool of 100 classifiers
pool_classifiers = BaggingClassifier(Perceptron(max_iter=10),
                                     n_estimators=100, random_state=rng)
pool_classifiers.fit(X_train, y_train)

# Calibrating Perceptrons to estimate probabilities
calibrated_pool = []
for clf in pool_classifiers:
    calibrated = CalibratedClassifierCV(base_estimator=clf, cv='prefit')
    calibrated.fit(X_dsel, y_dsel)
    calibrated_pool.append(calibrated)

# Initialize the DS techniques
knorau = KNORAU(calibrated_pool)
kne = KNORAE(calibrated_pool)
desp = DESP(calibrated_pool)
ola = OLA(calibrated_pool)
mcb = MCB(calibrated_pool, random_state=rng)
apriori = APriori(calibrated_pool, random_state=rng)
meta = METADES(calibrated_pool)

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
print('Classification accuracy KNORA-Union: ',
      knorau.score(X_test, y_test))
print('Classification accuracy KNORA-Eliminate: ',
      kne.score(X_test, y_test))
print('Classification accuracy DESP: ', desp.score(X_test, y_test))
print('Classification accuracy OLA: ', ola.score(X_test, y_test))
print('Classification accuracy A priori: ', apriori.score(X_test, y_test))
print('Classification accuracy MCB: ', mcb.score(X_test, y_test))
print('Classification accuracy META-DES: ', meta.score(X_test, y_test))
print('Classification accuracy Bagging: ',
      pool_classifiers.score(X_test, y_test))
