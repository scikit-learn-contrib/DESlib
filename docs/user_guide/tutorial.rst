.. _tutorial:

=========
Tutorial
=========

This tutorial will walk you through generating a pool of classifiers and applying several dynamic selection techniques
for the classification of unknown samples. The tutorial assumes that you are already familiar with the `Python language`_
and the `scikit-learn`_ library. Users not familiar with either Python and scikit-learn can start by checking out their tutorials.

Running Dynamic selection with Bagging
======================================

In this first tutorial, we do a step-by-step run of the example_bagging.py, that is included in the examples part of the DESlib.
This example uses the Wisconsin breast cancer dataset available on sklearn.datasets package.

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

The code starts by importing the corresponding DCS and DES algorithms that are tested as well as the other required
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

As DESlib is built on top of scikit-learn_, code will usually required the import of routines from it.

Preparing the dataset:
-----------------------

The next step is loading the data and dividing it into three partitions: Training, validation and test. In the dynamic
selection literature [1]_ the validation set is usually called the dynamic selection dataset (DSEL), since
this partition is used by the dynamic selection techniques in order to select the base classifiers, so in this
library we use the same terminology. The training set (X_train, y_train) is used to fit the pool of classifiers,
the validation (X_DSEL, y_DSEL) set is used to fit the dynamic selection models. The performance of the system
is then evaluated on the test set (X_test, y_test).

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


Another important aspect is to normalize the data so that it has
zero mean and unit variance, which is a common requirement for many machine learning algorithms.
This step can be easily done using the StandardScaler class from scikit-learn_. Note that the StandardScaler transform
should be fitted using the training and DSEL data only. Then, it can be applied for the test data.

An important point here is that in case of small datasets or when the base classifier models in the pool
are weak estimators such as Decision Stumps or Perceptrons, an overlap between the training data and DSEL
may be beneficial for achieving better performance.

Training a pool of classifiers:
-------------------------------

The next step is to generate a pool of classifiers. This list can be either
homogeneous (i.e., all base classifiers are of the same type) or heterogeneous (base classifiers of different types).
The library supports any type of base classifiers that is compatible with the scikit-learn library.

In this example, we generate a pool composed of 10 Perceptron classifiers
using the Bagging technique. It is important to mention that some DS techniques require that the base classifiers are capable of
estimating probabilities (i.e., implements the predict_proba function).

For the Perceptron model, this can be achieved
by calibrating the outputs of the base classifiers using the CalibratedClassifierCV class from scikit-learn.

.. code-block:: python

    model = CalibratedClassifierCV(Perceptron(max_iter=10))

    # Train a pool of 10 classifiers
    pool_classifiers = BaggingClassifier(model, n_estimators=10)
    pool_classifiers.fit(X_train, y_train)


Building the DS models
----------------------

Three DCS and four DES techniques are considered in this example:

- Overal Local Accuracy (OLA)
- Multiple-Classifier Behavior (MCB)
- A Priori selection
- K-Nearest Oracles-Union (KNU)
- K-Nearest Oracles-Eliminate (KNE)
- META-DES

**NEW:** Since version 0.3, DESlib does not require a trained pool of classifiers for instantiating its estimators. All estimator
can now be instantiated without specifying a pool of classifiers:

.. code-block:: python

    # DCS techniques
    ola = OLA()
    mcb = MCB()
    apriori = APriori()

    # DES techniques
    knorau = KNORAU()
    kne = KNORAE()
    desp = DESP()
    meta = METADES()

When the pool of classifiers is not specified, a standard :class:`BaggingClassifier` from sklearn is used, which generates
a pool composed of 10 decision trees. The parameter **DSEL_perc** controls the percentage of the input data that is used for fitting
DSEL. The remaining data will be used to fit the pool of classifiers. Note that this parameter is only taken into account if
the pool is either equals to None (when it was not informed) or still unfitted (when the base classifiers were not fitted)

However, since we already trained a pool of classifiers in the previous step we will continue this tutorial by instantiating the dynamic selection methods with an already fitted pool.
For more information on using DESlib estimators without specifying a trained pool of classifiers
see the  `examples page <auto_examples/index.html>`_.

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


Fitting the DS techniques:
---------------------------

The next step is to fit the DS model. We call the function fit to prepare the DS techniques for the
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

Note that if the pool of classifiers is still unfitted, this step will also fit the base classifiers in the pool.

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
Class labels and posterior probabilities can be easily calculated using the **predict** and **predict_proba** methods:

.. code-block:: python

    metades.predict(X_test)
    metades.predict_proba(X_test)

Changing parameters
-------------------

Changing the hyper-parameters is very easy. We just need to pass its value when instantiating a new method. For example,
we can change the size of the neighborhood used to estimate the competence level by setting the k value.

.. code-block:: python

    # DES techniques
    knorau = KNORAU(pool_classifiers, k=5)
    kne = KNORAE(pool_classifiers, k=5)

Also, we can change the mode DES algorithm works (dynamic selection, dynamic weighting or hybrid) by setting its mode:
.. code-block:: python

    meta = METADES(pool_classifiers, Hc=0.8, k=5, mode='hybrid')

In this code block, we change the size of the neighborhood (k=5), the value of the sample selection mechanism (Hc=0.8) and
also, state that the META-DES algorithm should work in a hybrid dynamic selection with and weighting mode.
The library accepts the change of several hyper-parameters. A list containing each one for all technique available
as well as its impact in the algorithm is presented in the `API Reference <api.html>`_.

Applying the Dynamic Frienemy Pruning (DFP)
-------------------------------------------

The library also implements the Dynamic Frienemy Pruning (DFP) proposed in [2]_. So any dynamic selection technique can be
applied using the FIRE (Frienemy Indecision Region Dynamic Ensemble Selection) framework. That is easily done by setting the
DFP to true when initializing a DS technique. In this example, we show how to start the FIRE-KNORA-U, FIRE-KNORA-E and FIRE-MCB techniques.

.. code-block:: python

    fire_knorau = KNORAU(pool_classifiers, DFP=True)
    fire_kne = KNORAE(pool_classifiers, DFP=True)
    fire_mcb = MCB(pool_classifiers, DFP=True)

We can also set the size of the neighborhood that is used to decide whether the query sample is located in a safe region or
in an indecision region (safe_k):

.. code-block:: python

    fire_knorau = KNORAU(pool_classifiers, DFP=True, safe_k=3)
    fire_kne = KNORAE(pool_classifiers, DFP=True, safe_k=5)
    fire_mcb = MCB(pool_classifiers, DFP=True, safe_k=7)

So, the fire_knorau will use a neighborhood composed of 3 samples, fire_knorae of 5 and fire_mcb of 7 in order to compute whether a given sample
is located in a indecision or safe region.

More tutorials on how to use different aspects of the library can be found in `examples page <auto_examples/index.html>`_

.. _Python language: https://docs.python.org/3.5/tutorial/
.. _scikit-learn: http://scikit-learn.org/stable/tutorial/index.html


References
-----------

.. [1] : R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,” Information Fusion, vol. 41, pp. 195 – 216, 2018.

.. [2] : Oliveira, D.V.R., Cavalcanti, G.D.C. and Sabourin, R., Online Pruning of Base Classifiers for Dynamic Ensemble Selection, Pattern Recognition, vol. 72, December 2017, pp 44-58.

