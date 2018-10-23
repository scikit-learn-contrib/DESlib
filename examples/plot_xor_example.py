# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause
"""
====================================================================
Dynamic selection with linear classifiers: XOR example
====================================================================

This example shows that DS can deal with non-linear problem (XOR) using
a combination of a few linear base classifiers.

- 6 dynamic selection methods (3 DES and 3 DCS) are evaluated with
  a pool composed of Decision stumps.


"""
###############################################################################
# Let's start by importing all required modules, and defining helper functions
# to facilitate plotting the decision boundaries:

import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from deslib.dcs.a_posteriori import APosteriori
from deslib.dcs.a_priori import APriori
from deslib.dcs.lca import LCA
from deslib.dcs.mcb import MCB
from deslib.dcs.mla import MLA
# DCS techniques
from deslib.dcs.ola import OLA
from deslib.dcs.rank import Rank
from deslib.des.des_clustering import DESClustering
from deslib.des.des_knn import DESKNN
# DES techniques
from deslib.des.des_p import DESP
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU
from deslib.des.meta_des import METADES
from deslib.util.datasets import make_xor


# Plotting-related functions
def plot_classifier_decision(ax, clf, X, mode='line', **params):

    xx, yy = make_grid(X[:, 0], X[:, 1])

    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    if mode == 'line':
        ax.contour(xx, yy, Z, **params)
    else:
        ax.contourf(xx, yy, Z, **params)
    ax.set_xlim((np.min(X[:, 0]), np.max(X[:, 0])))
    ax.set_ylim((np.min(X[:, 1]), np.max(X[:, 0])))


def plot_dataset(X, y, ax=None, title=None, **params):

    if ax is None:
        ax = plt.gca()
    ax.scatter(X[:, 0], X[:, 1], marker='o', c=y, s=25,
               edgecolor='k', **params)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    if title is not None:
        ax.set_title(title)
    return ax


def make_grid(x, y, h=.02):

    x_min, x_max = x.min() - 1, x.max() + 1
    y_min, y_max = y.min() - 1, y.max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


# Prepare the DS techniques. Changing k value to 5.
def initialize_ds(pool_classifiers, X, y, k=5):
    knorau = KNORAU(pool_classifiers, k=k)
    kne = KNORAE(pool_classifiers, k=k)
    desknn = DESKNN(pool_classifiers, k=k)
    ola = OLA(pool_classifiers, k=k)
    lca = LCA(pool_classifiers, k=k)
    mla = MLA(pool_classifiers, k=k)
    list_ds = [knorau, kne, ola, lca, mla, desknn]
    # fit the ds techniques
    for ds in list_ds:
        ds.fit(X, y)
    return list_ds


rng = np.random.RandomState(1234)
X, y = make_xor(1000, random_state=rng)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
X_DSEL, X_test, y_DSEL, y_test = train_test_split(X_test, y_test,
                                                  test_size=0.5)

pool_stumps = BaggingClassifier(DecisionTreeClassifier(max_depth=1),
                                n_estimators=100,
                                random_state=rng)
pool_stumps.fit(X_train, y_train)

list_ds_stumps = initialize_ds(pool_stumps, X_DSEL, y_DSEL)
for ds in list_ds_stumps:
    print('Accuracy ' + ds.name + ': ' + str(ds.score(X_test, y_test)))

###############################################################################
# Get the classification accuracy of the DS abd Bagging methods
# using the same pool of classifiers.

###############################################################################
# This example merge the training data with the validation, to create a
# DSEL having more examples for the competence estimation. Using the training
# data for dynamic selection can be beneficial  when dealing with small sample
# size datasets. However, in this case we need to have a pool composed of weak
# classifier so that the base classifiers are not able to memorize the
# training data (overfit).

X_DSEL = np.vstack((X_DSEL, X_train))
y_DSEL = np.hstack((y_DSEL, y_train))
list_ds_stumps = initialize_ds(pool_stumps, X_DSEL, y_DSEL, k=7)
for ds in list_ds_stumps:
    print('Accuracy ' + ds.name + ': ' + str(ds.score(X_test, y_test)))
print('Accuracy Bagging: ' + str(pool_stumps.score(X_test, y_test)))

# Plotting the decision border of the DS methods
for ds in list_ds_stumps:
    ax = plot_dataset(X, y)
    plot_classifier_decision(ax, ds, X_test)
    ax.set_xlim((np.min(X_test[:, 0]) - 0.1, np.max(X_test[:, 0] + 0.1)))
    ax.set_ylim((np.min(X_test[:, 1]) - 0.1, np.max(X_test[:, 1] + 0.1)))
    ax.set_title(ds.name)
plt.show()
