# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause
"""
====================================================================
Example DFP
====================================================================

In this example we show how to apply the dynamic frienemy pruning (DFP) to
different dynamic selection techniques.

The DFP method is an online pruning model which analyzes the region
of competence to know if it is composed of samples from different classes
(indecision region). Then, it remove the base classifiers that do not correctly
classifies at least a pair of samples coming from different classes
(i.e., the base classifiers that do not cross the local region.

The DFP is shown to significantly improve the performance of several
dynamic selection algorithms when dealing with heavily imbalanced problems,
as it avoids the classifiers that are biased towards the majority class
in predicting the label for the query.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from deslib.dcs import APosteriori
from deslib.dcs import APriori
from deslib.dcs import LCA
from deslib.dcs import OLA
from deslib.des import DESP
from deslib.des import METADES

rng = np.random.RandomState(654321)

# Generate a classification dataset
X, y = make_classification(n_classes=2, n_samples=2000, weights=[0.05, 0.95],
                           random_state=rng)
# split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state=rng)

# Split the data into training and DSEL for DS techniques
X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train,
                                                    test_size=0.5,
                                                    random_state=rng)
# Considering a pool composed of 10 base classifiers
pool_classifiers = RandomForestClassifier(n_estimators=10, random_state=rng,
                                          max_depth=10)
pool_classifiers.fit(X_train, y_train)

# DS techniques without DFP
apriori = APriori(pool_classifiers)
aposteriori = APosteriori(pool_classifiers)
ola = OLA(pool_classifiers)
lca = LCA(pool_classifiers)
desp = DESP(pool_classifiers)
meta = METADES(pool_classifiers)

apriori.fit(X_dsel, y_dsel)
aposteriori.fit(X_dsel, y_dsel)
ola.fit(X_dsel, y_dsel)
lca.fit(X_dsel, y_dsel)
desp.fit(X_dsel, y_dsel)
meta.fit(X_dsel, y_dsel)

###############################################################################
# Evaluating DES techniques
# ------------------------------
# Let's now evaluate the DES methods on the test set. Since we are dealing with
# imbalanced data, we use the AUC as performance metric instead of
# classification accuracy.

print('Evaluating DS techniques:')
print('F-Measure OLA: ', roc_auc_score(y_test, ola.predict(X_test)))
print('F-Measure LCA: ', roc_auc_score(y_test, lca.predict(X_test)))

print('F-Measure A priori: ', roc_auc_score(y_test, apriori.predict(X_test)))
print('Classification accuracy of A posteriori: ',
      roc_auc_score(y_test, aposteriori.predict(X_test)))
print('F-Measure DES-P: ', roc_auc_score(y_test, desp.predict(X_test)))
print('F-Measure META-DES: ', roc_auc_score(y_test, meta.predict(X_test)))

# Testing fire:
fire_apriori = APriori(pool_classifiers, DFP=True)
fire_aposteriori = APosteriori(pool_classifiers, DFP=True)
fire_ola = OLA(pool_classifiers, DFP=True)
fire_lca = LCA(pool_classifiers, DFP=True)
fire_desp = DESP(pool_classifiers, DFP=True)
fire_meta = METADES(pool_classifiers, DFP=True)

fire_apriori.fit(X_dsel, y_dsel)
fire_aposteriori.fit(X_dsel, y_dsel)
fire_ola.fit(X_dsel, y_dsel)
fire_lca.fit(X_dsel, y_dsel)
fire_desp.fit(X_dsel, y_dsel)
fire_meta.fit(X_dsel, y_dsel)

###############################################################################
# Evaluating FIRE-DES techniques
# ------------------------------
# Let's now evaluate the FIRE-DES methods based on AUC.

print('Evaluating FIRE-DS techniques:')
print('F-Measure OLA: ', roc_auc_score(y_test, fire_ola.predict(X_test)))
print('F-Measure LCA: ', roc_auc_score(y_test, fire_lca.predict(X_test)))

print('F-Measure A priori: ', roc_auc_score(y_test, fire_apriori.predict(X_test)))
print('Classification accuracy of A posteriori: ',
      roc_auc_score(y_test, fire_aposteriori.predict(X_test)))
print('F-Measure DES-P: ', roc_auc_score(y_test, fire_desp.predict(X_test)))
print('F-Measure META-DES: ', roc_auc_score(y_test, fire_meta.predict(X_test)))