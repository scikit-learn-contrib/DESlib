Welcome to DESlib documentation!
=================================================

DESlib is an ensemble learning library focusing the implementation of the state-of-the-art techniques for dynamic classifier and ensemble selection.

DESlib is a work in progress. Contributions are welcomed through its GitHub page: https://github.com/Menelau/DESlib.

Introduction:
-------------

Dynamic Selection (DS) refers to techniques in which the base classifiers are selected
on the fly, according to each new sample to be classified. Only the most competent, or an ensemble containing the most competent classifiers is selected to predict
the label of a specific test sample. The rationale for such techniques is that not every classifier in
the pool is an expert in classifying all unknown samples; rather, each base classifier is an expert in
a different local region of the feature space.

DS is one of the most promising MCS approaches due to the fact that
more and more works are reporting the superior performance of such techniques over static combination methods. Such techniques
have achieved better classification performance especially when dealing with small-sized and imbalanced datasets.


Installation:
-------------

The package can be installed using pip:

Stable version:

.. code-block:: bash

    pip install deslib

Latest version (under development):

.. code-block:: bash

    pip install git+https://github.com/Menelau/DESlib

DESlib is tested to work with Python 3.5, and 3.6. The dependency requirements are:

* scipy(>=0.13.3)
* numpy(>=1.10.4)
* scikit-learn(>=0.19.0)

These dependencies are automaticatically installed using the pip commands above.

API Reference:
--------------
.. toctree::
    :maxdepth: 3

    modules/des
    modules/dcs
    modules/static
    modules/util

Examples:
---------

Example using the KNORA-E techniques using a random forest to generate the pool of classifiers:

.. code-block:: python

    from sklearn.ensemble import RandomForestClassifier
    from deslib.des.knora_e import KNORAE


    # Train a pool of 10 classifiers
    pool_classifiers = RandomForestClassifier(n_estimators=10)
    pool_classifiers.fit(X_train, y_train)

    # Initialize the DES model
    knorae = KNORAE(pool_classifiers)

    # Preprocess the Dynamic Selection dataset (DSEL)
    knorae.fit(X_dsel, y_dsel)

    # Predict new examples:
    knorae.predict(X_test)

The library accepts any list of classifiers (from scikit-learn) as input, including a list containing different classifier models (heterogeneous ensembles).
More examples to use the API can be found in the documentation and im the Examples directory.

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

.. _GitHub: https://github.com/Menelau/DESlib
