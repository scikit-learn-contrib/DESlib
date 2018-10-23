# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np
import matplotlib.pyplot as plt
# DCS techniques
from deslib.dcs.mcb import MCB
from deslib.dcs.ola import OLA
from deslib.dcs.rank import Rank

# DES techniques
from deslib.des.des_p import DESP
from deslib.des.knora_e import KNORAE
from deslib.des.knora_u import KNORAU

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = load_breast_cancer()
X = dataset.data
y = dataset.target

X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler = StandardScaler()
X_train = scaler.fit_transform(X)
X_test = scaler.transform(X_test)

# Since version 0.3, the pool of classifiers can be fitted inside the DS
# method
mcb = MCB(with_IH=True).fit(X_train, y_train)
ola = OLA(with_IH=True).fit(X_train, y_train)
rank = Rank(with_IH=True).fit(X_train, y_train)
des_p = DESP(with_IH=True).fit(X_train, y_train)
kne = KNORAE(with_IH=True).fit(X_train, y_train)
knu = KNORAU(with_IH=True).fit(X_train, y_train)
list_ds_methods = [mcb, ola, rank, des_p, kne, knu]
list_ih_values = [0.0, 0.14, 0.28, 0.42]

accuracy = np.zeros((len(list_ds_methods), len(list_ih_values)))
for idx_ih, ih_rate in [0.0, 0.14, 0.28, 0.42]:
    for idx_ds, ds_method in enumerate(list_ds_methods):
        ds_method.__setattr__("IH_rate", ih_rate)
        accuracy[idx_ds, idx_ih] = ds_method.score(X_test, y_test)

# Plot accuracy x IH
print(accuracy)



