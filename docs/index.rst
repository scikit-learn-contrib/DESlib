Welcome to DESlib documentation!
=================================================

DESlib is an ensemble learning library focusing the implementation of the state-of-the-art techniques for dynamic classifier
and ensemble selection.

DESlib is a work in progress. Contributions are welcomed through its GitHub page: https://github.com/Menelau/DESlib.

Introduction
--------------
Dynamic Selection (DS) refers to techniques in which the base classifiers are selected
on the fly, according to each new sample to be classified. Only the most competent, or an ensemble containing the most competent classifiers is selected to predict
the label of a specific test sample. The rationale for such techniques is that not every classifier in
the pool is an expert in classifying all unknown samples; rather, each base classifier is an expert in
a different local region of the feature space.

DS is one of the most promising MCS approaches due to the fact that
more and more works are reporting the superior performance of such techniques over static combination methods. Such techniques
have achieved better classification performance especially when dealing with small-sized and imbalanced datasets. A
comprehensive review of dynamic selection can be found in the following papers [1]_ [2]_

Philosophy
-----------
DESlib was developed with two objectives in mind: to make it easy to integrate Dynamic Selection algorithms to
machine learning projects, and to facilitate research on this topic, by providing implementations of the main
DES and DCS methods, as well as the commonly used baseline methods. Each algorithm implements the main methods
in the scikit-learn_ API **scikit-learn**: **fit(X, y)**, **predict(X)**, **predict_proba(X)**
and **score(X, y)**.

The implementation of the DS methods is modular, following a taxonomy defined in [1]_.
This taxonomy considers the main characteristics of DS methods, that are centered in three components:

1. the methodology used to define the local region, in which the competence level of the base classifiers are estimated (region of competence);
2. the source of information used to estimate the competence level of the base classifiers.
3. the selection approach to define the best classifier (for DCS) or the best set of classifiers (for DES).

This modular approach makes it easy for researchers to implement new DS methods, in many cases requiring only the
implementation of the method **estimate_competence**, that is, how the local competence of the base classifier is measured.

`API Reference <api.html>`_
----------------------------

If you are looking for information on a specific function, class or
method, this part of the documentation is for you.

.. toctree::
    :hidden:

    user_guide
    api
    auto_examples/index
    news


`Example <auto_examples/index.html>`_
----------------------------------------

Here we present an example of the KNORA-E techniques using a random forest to generate the pool of classifiers:

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
More examples to use the API can be found in the `examples page <auto_examples/index.html>`_.


Citation
==================

If you use DESLib in a scientific paper, please consider citing the following paper:

Rafael M. O. Cruz, Luiz G. Hafemann, Robert Sabourin and George D. C. Cavalcanti **DESlib: A Dynamic ensemble selection library in Python.** arXiv preprint arXiv:1802.04967 (2018).

.. code-block:: text

    @article{cruz_deslib:2018,
        title = {{DESlib}: {A} {Dynamic} ensemble selection library in {Python}},
        journal = {arXiv preprint arXiv:1802.04967},
        author = {Cruz, Rafael M. O. and Hafemann, Luiz G. and Sabourin, Robert and Cavalcanti, George D. C.},
        year = {2018}
    }



References
-----------
.. [1] : R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,” Information Fusion, vol. 41, pp. 195 – 216, 2018.

.. [2] : A. S. Britto, R. Sabourin, L. E. S. de Oliveira, Dynamic selection of classifiers - A comprehensive review, Pattern Recognition 47 (11) (2014) 3665–3680.

.. _scikit-learn: http://scikit-learn.org/stable/

.. _GitHub: https://github.com/Menelau/DESlib
