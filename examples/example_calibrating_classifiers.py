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

###############################################################################
# Preparing the dataset
# ---------------------
# In this part we load the breast cancer dataset from scikit-learn and
# preprocess it in order to pass to the DS models. An important point here is
# to normalize the data so that it has zero mean and unit variance, which is
# a common requirement for many machine learning algorithms.
# This step can be easily done using the StandardScaler class.

rng = np.random.RandomState(123)
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

# Train a pool of 100 base classifiers
pool_classifiers = BaggingClassifier(Perceptron(max_iter=10),
                                     n_estimators=100, random_state=rng)
pool_classifiers.fit(X_train, y_train)

# Initialize the DS techniques
knorau = KNORAU(pool_classifiers)
kne = KNORAE(pool_classifiers)
desp = DESP(pool_classifiers)
ola = OLA(pool_classifiers)
mcb = MCB(pool_classifiers, random_state=rng)

###############################################################################
# Calibrating base classifiers
# -----------------------------
# Some dynamic selection techniques requires that the base classifiers estimate
# probabilities in order to estimate its competence level. Since the Perceptron
# model is not a probabilistic classifier (does not implements the
# predict_proba method, it needs to be calibrated for
# probability estimation before being used by such DS techniques. This step can
# be conducted using the CalibrateClassifierCV class from scikit-learn. Note
# that in this example we pass a prefited pool of classifiers to the
# calibration method in order to use exactly the same pool used in the other
# DS methods.
calibrated_pool = []
for clf in pool_classifiers:
    calibrated = CalibratedClassifierCV(base_estimator=clf, cv='prefit')
    calibrated.fit(X_dsel, y_dsel)
    calibrated_pool.append(calibrated)

apriori = APriori(calibrated_pool, random_state=rng)
meta = METADES(calibrated_pool)


knorau.fit(X_dsel, y_dsel)
kne.fit(X_dsel, y_dsel)
desp.fit(X_dsel, y_dsel)
ola.fit(X_dsel, y_dsel)
mcb.fit(X_dsel, y_dsel)
apriori.fit(X_dsel, y_dsel)
meta.fit(X_dsel, y_dsel)

###############################################################################
# Evaluating the methods
# -----------------------
# Let's now evaluate the methods on the test set. We also use the performance
# of Bagging (pool of classifiers without any selection) as a baseline
# comparison. We can see that  the majority of DS methods achieve higher
# classification accuracy.

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
