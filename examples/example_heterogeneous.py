"""
====================================================================
Example using heterogeneous ensemble
====================================================================
DESlib accepts different classifier models in the pool of classifiers.
Such pool of classifiers is called Heterogeneous.

In this example, we consider a pool of classifiers composed of a
Gaussian Naive Bayes, Perceptron, k-NN, Decision tree and Gaussian SVM. We
also compare the result of DS methods with the voting classifier from sklearn.
"""
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
# Importing dataset and preprocessing routines
from sklearn.datasets import fetch_openml
from sklearn.ensemble import VotingClassifier
# Base classifier models:
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from deslib.dcs import MCB
# Example of DCS techniques
from deslib.dcs import OLA
from deslib.des import DESP
# Example of DES techniques
from deslib.des import KNORAE
from deslib.des import KNORAU
from deslib.des import METADES
from deslib.static import StackedClassifier

rng = np.random.RandomState(42)
data = fetch_openml(name='phoneme', cache=False, as_frame=False)
X = data.data
y = data.target

# split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state=rng)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Split the data into training and DSEL for DS techniques
X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train,
                                                    test_size=0.5,
                                                    random_state=rng)

model_perceptron = CalibratedClassifierCV(Perceptron(max_iter=100,
                                                     random_state=rng),
                                          cv=3)

model_perceptron.fit(X_train, y_train)
model_svc = SVC(probability=True, gamma='auto',
                random_state=rng).fit(X_train, y_train)
model_bayes = GaussianNB().fit(X_train, y_train)
model_tree = DecisionTreeClassifier(random_state=rng,
                                    max_depth=10).fit(X_train, y_train)
model_knn = KNeighborsClassifier(n_neighbors=7).fit(X_train, y_train)

pool_classifiers = [model_perceptron,
                    model_svc,
                    model_bayes,
                    model_tree,
                    model_knn]

voting_classifiers = [("perceptron", model_perceptron),
                      ("svc", model_svc),
                      ("bayes", model_bayes),
                      ("tree", model_tree),
                      ("knn", model_knn)]

model_voting = VotingClassifier(estimators=voting_classifiers).fit(
    X_train, y_train)

# Initializing the techniques
knorau = KNORAU(pool_classifiers)
kne = KNORAE(pool_classifiers)
desp = DESP(pool_classifiers)
metades = METADES(pool_classifiers)
# DCS techniques
ola = OLA(pool_classifiers)
mcb = MCB(pool_classifiers)

##############################################################################
# Adding stacked classifier as baseline comparison. Stacked classifier can
# be found in the static module. In this experiment we consider two types
# of stacking: one using logistic regression as meta-classifier
# (default configuration) and the other using a Decision Tree.
stacked_lr = StackedClassifier(pool_classifiers, random_state=rng)
stacked_dt = StackedClassifier(pool_classifiers,
                               random_state=rng,
                               meta_classifier=DecisionTreeClassifier())
# Fitting the DS techniques
knorau.fit(X_dsel, y_dsel)
kne.fit(X_dsel, y_dsel)
desp.fit(X_dsel, y_dsel)
metades.fit(X_dsel, y_dsel)
ola.fit(X_dsel, y_dsel)
mcb.fit(X_dsel, y_dsel)

# Fitting the tacking models
stacked_lr.fit(X_dsel, y_dsel)
stacked_dt.fit(X_dsel, y_dsel)

# Calculate classification accuracy of each technique
print('Evaluating DS techniques:')
print('Classification accuracy of Majority voting the pool: ',
      model_voting.score(X_test, y_test))
print('Classification accuracy of KNORA-U: ', knorau.score(X_test, y_test))
print('Classification accuracy of KNORA-E: ', kne.score(X_test, y_test))
print('Classification accuracy of DESP: ', desp.score(X_test, y_test))
print('Classification accuracy of META-DES: ', metades.score(X_test, y_test))
print('Classification accuracy of OLA: ', ola.score(X_test, y_test))
print('Classification accuracy Stacking LR', stacked_lr.score(X_test, y_test))
print('Classification accuracy Stacking DT', stacked_dt.score(X_test, y_test))
