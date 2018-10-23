# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause
"""
====================================================================
Dynamic selection vs K-NN: Using instance hardness
====================================================================

One aspect about dynamic selection techniques is that it can better deal with
the classification of test examples associated with high degree of instance
hardness. Such examples are often found close to the border of the classes,
with the majority of its neighbors belongind to different classes.

DESlib already implements a switch mechanism between DS techniques and the KNN
classifier according to the hardness level of an instance. This example
varies the threshold in which KNN is used for classification instead of DS
methods. It also compares the classification results with the standard KNN
as a baseline.

The switch mechanism also reduces the computational cost involved since part
of the samples are classified by the DS method.

"""

###############################################################################
# Let's start by importing all required modules. In this example we use the
# new sklearn-OpenML interface to fetch a different classification problem.
import numpy as np
import matplotlib.pyplot as plt
# DCS techniques
from deslib.dcs import MCB
from deslib.dcs import OLA
from deslib.dcs import Rank
from deslib.dcs import LCA

# DES techniques
from deslib.des import DESP
from deslib.des import KNORAU
from deslib.des import KNORAE

from sklearn.model_selection import train_test_split
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

rng = np.random.RandomState(123456)

data = fetch_openml(name='diabetes')
X = data.data
y = data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=rng)

# Normalizing the dataset to have 0 mean and unit variance.
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Training a pool of classifiers using the bagging technique.
pool_classifiers = BaggingClassifier(Perceptron(), random_state=rng)
pool_classifiers.fit(X_train, y_train)

# Setting with_IH
mcb = MCB(pool_classifiers)
ola = OLA(pool_classifiers)
des_p = DESP(pool_classifiers)
knu = KNORAU(pool_classifiers)
lca = LCA(pool_classifiers)
kne = KNORAE(pool_classifiers)
rank = Rank(pool_classifiers)
list_ds_methods = [mcb, ola, des_p, knu, lca, kne, rank]
names = ['MCB', 'OLA', 'DES-P', 'KNORA-U', 'LCA', 'KNORA-E', 'Rank']

k_value_list = range(3, 15)
# Plot accuracy x IH
fig, ax = plt.subplots()
for ds_method, name in zip(list_ds_methods, names):
    accuracy = []
    for k in k_value_list:
        ds_method.k = k
        ds_method.fit(X_train, y_train)
        accuracy.append(ds_method.score(X_test, y_test))
    ax.plot(k_value_list, accuracy, label=name)

plt.xticks(k_value_list)
ax.set_ylim(0.63, 0.80)
ax.set_xlabel('K value', fontsize=13)
ax.set_ylabel('Accuracy on the test set (%)', fontsize=13)
ax.legend()
plt.show()
