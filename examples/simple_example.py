# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause
"""
====================================================================
Simple example
====================================================================

In this example we show how to apply different DCS and DES techniques for a
classification dataset.

"""
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from deslib.des import METADES
from deslib.des import KNORAE


# Setting up the random state to have consistent results
rng = np.random.RandomState(42)

# Generate a classification dataset
X, y = make_classification(n_samples=1000, random_state=rng)
# split the data into training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,
                                                    random_state=rng)

# Split the data into training and DSEL for DS techniques
X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train,
                                                    test_size=0.5,
                                                    random_state=rng)

# Initialize the DS techniques. DS methods can be initialized without
# specifying a single input parameter. In this example, we just pass the random
# state in order to always have the same result.
kne = KNORAE(random_state=rng)
meta = METADES(random_state=rng)

# Fitting the des techniques
kne.fit(X_dsel, y_dsel)
meta.fit(X_dsel, y_dsel)

# Calculate classification accuracy of each technique
print('Evaluating DS techniques:')
print('Classification accuracy KNORA-Eliminate: ',
      kne.score(X_test, y_test))
print('Classification accuracy META-DES: ', meta.score(X_test, y_test))
