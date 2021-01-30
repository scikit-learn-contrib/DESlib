# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause
"""
====================================================================
Measuring the influence of the region of competence
====================================================================

This example shows how the size of the region of competence (parameter k)
can influence the final performance of DS techniques.

In this example we vary the value of the parameter k from 3 to 15 and measure
the performance of 7 different dynamic selection technique using the same
pool of classifiers.

"""

###############################################################################
# Let's start by importing all required modules. In this example we use the
# new sklearn-OpenML interface to fetch the diabetes classification problem.

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from deslib.dcs import LCA
# DCS techniques
from deslib.dcs import MCB
from deslib.dcs import OLA
from deslib.dcs import Rank
# DES techniques
from deslib.des import DESP
from deslib.des import KNORAE
from deslib.des import KNORAU

rng = np.random.RandomState(123456)

data = fetch_openml(name='diabetes', cache=False, as_frame=False)
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)

# Normalizing the dataset to have 0 mean and unit variance.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

pool_classifiers = BaggingClassifier(Perceptron(max_iter=100),
                                     random_state=rng)
pool_classifiers.fit(X_train, y_train)

# Setting with_IH
mcb = MCB(pool_classifiers, random_state=rng)
ola = OLA(pool_classifiers)
des_p = DESP(pool_classifiers)
knu = KNORAU(pool_classifiers)
lca = LCA(pool_classifiers)
kne = KNORAE(pool_classifiers)
rank = Rank(pool_classifiers)
list_ds_methods = [mcb, ola, des_p, knu, lca, kne, rank]
names = ['MCB', 'OLA', 'DES-P', 'KNORA-U', 'LCA', 'KNORA-E', 'Rank']

k_value_list = range(3, 16)

###############################################################################
# Plot accuracy x region of competence size.
# -------------------------------------------
# We can see the this parameter can have a huge influence in the performance
# of certain DS techniques. The main exception being the KNORA-E and Rank
# which have built-in mechanism to automatically adjust the region
# of competence size during the competence level estimation.

fig, ax = plt.subplots()
for ds_method, name in zip(list_ds_methods, names):
    accuracy = []
    for k in k_value_list:
        ds_method.k = k
        ds_method.fit(X_train, y_train)
        accuracy.append(ds_method.score(X_test, y_test))
    ax.plot(k_value_list, accuracy, label=name)

plt.xticks(k_value_list)
ax.set_ylim(0.60, 0.80)
ax.set_xlabel('Region of competence size (K value)', fontsize=13)
ax.set_ylabel('Accuracy on the test set (%)', fontsize=13)
ax.legend(loc='lower right')
plt.show()
