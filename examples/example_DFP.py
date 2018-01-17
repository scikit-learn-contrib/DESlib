"""
In this example we show how to apply the dynamic frienemy pruning (DFP) to different dynamic selection techniques.

The DFP method is an online pruning model which analyzes the region of competence to know if it is composed of samples
from different classes (indecision region). Then, it remove the base classifiers that do not correctly classifies
at least a pair of samples coming from different classes (i.e., the base classifiers that do not cross the local
region.

The DFP is shown to significantly improve the performance of several dynamic selection algorithms when dealing
with heavily imbalanced problems, as it avoids the classifiers that are biased towards the majority class in predicting
the label for the query.
"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Example of dcs techniques:
from deslib.dcs.a_posteriori import APosteriori
from deslib.dcs.lca import LCA
from deslib.dcs.ola import OLA
from deslib.dcs.a_priori import APriori

# Example of des techniques:
from deslib.des.meta_des import METADES
from deslib.des.des_p import DESP

rng = np.random.RandomState(654321)

# Generate a classification dataset
X, y = make_classification(n_classes=2, n_samples=1000, weights=[0.2, 0.8], random_state=rng)
# split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=rng)


# Split the data into training and DSEL for DS techniques
X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=0.5, random_state=rng)
# Considering a pool composed of 10 base classifiers
pool_classifiers = RandomForestClassifier(n_estimators=10, random_state=rng, max_depth=10)
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

print('Evaluating DS techniques:')
print('Classification accuracy of OLA: ', ola.score(X_test, y_test))
print('Classification accuracy of LCA: ', lca.score(X_test, y_test))
print('Classification accuracy of A priori: ', apriori.score(X_test, y_test))
print('Classification accuracy of A posteriori: ', aposteriori.score(X_test, y_test))
print('Classification accuracy of DES-P: ', desp.score(X_test, y_test))
print('Classification accuracy of META-DES: ', meta.score(X_test, y_test))

# Testing fire:
fire_apriori = APriori(pool_classifiers, DFP=True)
fire_aposteriori = APosteriori(pool_classifiers, DFP=True)
fire_ola = OLA(pool_classifiers, DFP=True)
fire_lca = LCA(pool_classifiers, DFP=True)
fire_desp = DESP(pool_classifiers, DFP=True)
fire_meta = METADES(pool_classifiers, DFP=True)

fire_apriori.fit(X_dsel, y_dsel)
fire_aposteriori.fit(X_dsel, y_dsel)
fire_ola .fit(X_dsel, y_dsel)
fire_lca.fit(X_dsel, y_dsel)
fire_desp.fit(X_dsel, y_dsel)
fire_meta.fit(X_dsel, y_dsel)

print('Evaluating FIRE-DS techniques:')
print('Classification accuracy of FIRE-OLA: ', fire_ola.score(X_test, y_test))
print('Classification accuracy of FIRE-LCA: ', fire_lca.score(X_test, y_test))
print('Classification accuracy of FIRE-A priori: ', fire_apriori.score(X_test, y_test))
print('Classification accuracy of FIRE-A posteriori: ', aposteriori.score(X_test, y_test))
print('Classification accuracy of FIRE-DES-P: ', fire_desp.score(X_test, y_test))
print('Classification accuracy of FIRE-META-DES: ', fire_meta.score(X_test, y_test))
