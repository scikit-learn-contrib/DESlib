.. _tutorial:

=========
Tutorial
=========

This tutorial will walk you through generating a pool of classifiers and applying several dynamic selection techniques
for the classification of unknown samples. The tutorial assumes that you are already familiar with the `Python language`_
and the `scikit-learn`_ library. Users not familiar with either Python and scikit-learn can start by checking out their tutorials.

Running Dynamic selection with Bagging
======================================

In this first tutorial, we do a step-by-step run of the example_bagging.py that is included in the DESlib.

The first step is to run the example to check if everything is working as intended:

.. code-block:: bash

    cd examples
    python example_bagging.py

This script run six different dynamic selection models: Three DCS (OLA, A-Priori, MCB) and four DES (KNORA-Union,
KNORA-Eliminate, DES-P and META-DES)
The example outputs the classification accuracy of each dataset:

.. code-block:: text

  Evaluating DS techniques:
  Classification accuracy KNORA-Union:  0.973404255319
  Classification accuracy KNORA-Eliminate:  0.968085106383
  Classification accuracy DESP:  0.973404255319
  Classification accuracy OLA:  0.968085106383
  Classification accuracy A priori:  0.973404255319
  Classification accuracy MCB:  0.968085106383
  Classification accuracy META-DES:  0.973404255319

Code analysis:
==============

The first thing to do is to import the corresponding DCS and DES algorithms that are tested as well as the other required
libraries:

.. code-block:: python

    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import Perceptron
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import BaggingClassifier

    #importing DCS techniques from DESlib
    from deslib.dcs.ola import OLA
    from deslib.dcs.a_priori import APriori
    from deslib.dcs.mcb import MCB

    #import DES techniques from DESlib
    from deslib.des.des_p import DESP
    from deslib.des.knora_u import KNORAU
    from deslib.des.knora_e import KNORAE
    from deslib.des.meta_des import METADES

As DESlib is built on top of scikit-learn classifier, code will usually required the import of routines from this library.

Preaparing the dataset:
-----------------------

Before exploiting the models we need to prepare the dataset. We use the breast cancer dataset from scikit learn. Also we divide
it into three partitions: Training, Test, and Dynamic Selection (DSEL). In the dynamic selection library, we usually call the dataset
that is used for the competence level estimation as the dynamic selection dataset (DSEL) rather than the validation dataset.

.. code-block:: python

    data = load_breast_cancer()
    X = data.data
    y = data.target
    # split the data into training and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

    # Scale the variables to have 0 mean and unit variance
    scalar = StandardScaler()
    X_train = scalar.fit_transform(X_train)
    X_test = scalar.transform(X_test)

    # Split the data into training and DSEL for DS techniques
    X_train, X_dsel, y_train, y_dsel = train_test_split(X_train, y_train, test_size=0.5)

There are two important points in the dataset preparation:

The features must be normalized to have 0 mean and unit variance. This normalization helps in the training steps of the classifiers pool
as well as for the calculations of the distance between the query sample and its nearest neighbors.

The second important point is that, in the current version, the library only accepts class labels
starting from 0 to L-1 (where L is the number of classes). If your dataset does not follow this requirement, you can use the
LabelEncoder class form scikit-learn to prepare the data. As the datasets loaded from scikit-learn already follow this rule, we can
skip this step.

Training a pool of classifiers:
-------------------------------

Here we generate a pool of classifiers using the Bagging technique. The Perceptron classifier is used as the base classifier model
in this example. It is important to mention that some DS techniques requires that the base classifiers are capable of
estimating probabilities. For the Perceptron model, this can be achieved by calibrating the outputs of the base
classifiers using the CalibratedClassifierCV class from scikit-learn.

.. code-block:: python

    model = CalibratedClassifierCV(Perceptron(max_iter=10))

    # Train a pool of 10 classifiers
    pool_classifiers = BaggingClassifier(model, n_estimators=10)
    pool_classifiers.fit(X_train, y_train)


Building the DS models
----------------------
Initializing DS techniques
Here we initialize the DS techniques. Three DCS and three DES techniques are considered in this example.
The only parameter that is required by the techniques is the pool of classifiers. All others are optional parameters
which can be specified.


.. code-block:: python

    # DCS techniques
    ola = OLA(pool_classifiers)
    mcb = MCB(pool_classifiers)
    apriori = APriori(pool_classifiers)

    # DES techniques
    knorau = KNORAU(pool_classifiers)
    kne = KNORAE(pool_classifiers)
    desp = DESP(pool_classifiers)
    meta = METADES(pool_classifiers)

Changing parameters
^^^^^^^^^^^^^^^^^^^^
Changing the hyper-parameters is very easy. We just need to pass its value when instantiating a new method. For example
in we can change the size of the neighborhood used to estimate the competence level by simply .

.. code-block:: python

    # DES techniques
    knorau = KNORAU(pool_classifiers, k=5)
    kne = KNORAE(pool_classifiers, k=5)

Also we can change the mode DES algorithm works (dynamic selection, dynamic weighting or hybrid) by setting its mode:
.. code-block:: python

    meta = METADES(pool_classifiers, Hc=0.8, k=5, mode='hybrid')

In this code block, we change the size of the neighborhood (k=5), the value of the sample selection mechanism (Hc=0.8) and
also state that the META-DES algorithm should work in a hybrid dynamic selection with and weighting mode.

The library accepts the change of several hyper-parameters. A list containing each one for all technique available
as well as its impact in the algorithm is presented in the API Reference.

Applying the Dynamic Frienemy Pruning (DFP)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: python

    knorau = KNORAU(pool_classifiers, DFP=True)
    kne = KNORAE(pool_classifiers, DFP=True)
    mcb = MCB(pool_classifiers, DFP=True)

We can also set the size of the neighborhood that is used to decide whether the query sample is located in a safe region or
in an indecision region (safe_k):

.. code-block:: python

    knorau = KNORAU(pool_classifiers, DFP=True, safe_k=5)
    kne = KNORAE(pool_classifiers, DFP=True, safe_k=5)
    mcb = MCB(pool_classifiers, DFP=True, safe_k=5)

Fitting the DS techniques:
---------------------------

After method is initialized, we need to fit the DS model. We call the function fit to prepare the DS techniques for the
classification of new data by pre-processing the information required to apply the DS techniques, such as,
fitting the algorithm used to estimate the region of competence (k-NN, k-Means) and calculating the source of competence
of the base classifiers for each sample in the dynamic selection dataset.

.. code-block:: python

    knorau.fit(X_dsel, y_dsel)
    kne.fit(X_dsel, y_dsel)
    desp.fit(X_dsel, y_dsel)
    ola.fit(X_dsel, y_dsel)
    mcb.fit(X_dsel, y_dsel)
    apriori.fit(X_dsel, y_dsel)
    meta.fit(X_dsel, y_dsel)

Estimating classification accuracy:
------------------------------------
Estimating the classification accuracy of each method is very easy. Each DS technique implements the function score
from scikit-learn in order to estimate the classification accuracy.

.. code-block:: python

    print('Classification accuracy OLA: ', ola.score(X_test, y_test))
    print('Classification accuracy A priori: ', apriori.score(X_test, y_test))
    print('Classification accuracy KNORA-Union: ', knorau.score(X_test, y_test))
    print('Classification accuracy KNORA-Eliminate: ', kne.score(X_test, y_test))
    print('Classification accuracy DESP: ', desp.score(X_test, y_test))
    print('Classification accuracy META-DES: ', apriori.score(X_test, y_test))

However, you may need to calculate the predictions of the model or the estimation of probabilities instead of only computing the accuracy.
These information can be easily calculated using the **predict** and **predict_proba** methods:

.. code-block:: python

    metades.predict(X_test)
    metades.predict_proba(X_test)

For more examples you can consult as well as on ipython notebooks page:

.. _Python language: https://docs.python.org/3.5/tutorial/
.. _scikit-learn: http://scikit-learn.org/stable/tutorial/index.html
