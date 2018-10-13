# coding: utf-8
"""
====================================================================
Dynamic selection on non-linear problems (XOR example)
====================================================================

This example shows that DS can deal with non-linear problem (XOR) using
a combination of a few linear base classifiers.

- 14 dynamic selection methods (7 DES and 7 DCS) are evaluated with
a pool composed of either Perceptrons or Decision stumps as base classifiers.

- This example also compares the performance of Bagging and Boosting (AdaBoost)
techniques, showing that they fail to properly solve this problem using
only linear classifiers.

"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from syndata.plot_tools import plot_classifier_decision, plot_dataset

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

# ## Create the XOR problem with 1000 examples and plot its distribution


X, y = make_xor(1000)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5)
X_DSEL, X_test, y_DSEL, y_test = train_test_split(X_test, y_test,
                                                  test_size=0.5)
axdata = plot_dataset(X, y)
axdata.axvline(x=0.5, color='k')
axdata.axhline(y=0.5, color='k')

# Train the pool of classifiers
pool_perceptron = BaggingClassifier(Perceptron(max_iter=5),
                                    n_estimators=100).fit(X_train, y_train)
pool_stumps = BaggingClassifier(DecisionTreeClassifier(max_depth=1),
                                n_estimators=100).fit(X_train, y_train)

# Calibrate pool for probabilities estimates

pool_perceptron_calibrated = []
for clf in pool_perceptron:
    calibrated_clf = CalibratedClassifierCV(clf, cv='prefit')
    calibrated_clf.fit(X_DSEL, y_DSEL)
    pool_perceptron_calibrated.append(calibrated_clf)


# Prepare the DS techniques
# 
# Initialize all DS techniques. Since all DS methods have the same
# signature,  we can easily create a list containing
# all of them to evaluate the performance later.
def initialize_ds(pool_classifiers, X, y, k=7):
    knorau = KNORAU(pool_classifiers, k=k)
    kne = KNORAE(pool_classifiers, k=k)
    desknn = DESKNN(pool_classifiers, k=k)
    ola = OLA(pool_classifiers, k=k)
    lca = LCA(pool_classifiers, k=k)
    mla = MLA(pool_classifiers, k=k)
    mcb = MCB(pool_classifiers, k=k)
    desp = DESP(pool_classifiers, k=k)
    rank = Rank(pool_classifiers, k=k)
    apri = APriori(pool_classifiers, k=k)
    apos = APosteriori(pool_classifiers, k=k)
    des_clustering = DESClustering(pool_classifiers)
    metades = METADES(pool_classifiers, k=k)

    list_ds = [knorau, kne, ola, lca, mcb, desp, rank, apri, apos,
               des_clustering, metades, mla, desknn]

    # fit the ds techniques
    for ds in list_ds:
        ds.fit(X, y)
    return list_ds


list_ds_stumps = initialize_ds(pool_stumps, X_DSEL, y_DSEL)
for ds in list_ds_stumps:
    print('Accuracy ' + ds.name + ': ' + str(ds.score(X_test, y_test)))

# Get the classification accuracy of the DS abd Bagging methods
# using the same pool of classifiers.
#
# This example merge the training data with the validation, to create a
# DSEL having more examples for the competence estimation. Using the training
# data for dynamic selection can be beneficial  when dealing with small sample
# size datasets. However, in this case we need to have a pool composed of weak
# classifier so that the base classifiers are not able to memorize the
# training data (overfit).

X_DSEL = np.vstack((X_DSEL, X_train))
y_DSEL = np.hstack((y_DSEL, y_train))
list_ds_stumps = initialize_ds(pool_stumps, X_DSEL, y_DSEL, k=10)
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
