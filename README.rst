.. image:: https://readthedocs.org/projects/deslib/badge/?version=latest
    :target: http://deslib.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

.. image:: https://travis-ci.org/Menelau/DESlib.svg?branch=master
    :target: https://travis-ci.org/Menelau/DESlib

.. image:: https://codecov.io/gh/Menelau/DESlib/branch/master/graph/badge.svg
    :target: https://codecov.io/gh/Menelau/DESlib

.. image:: https://landscape.io/github/Menelau/DESlib/master/landscape.svg?style=flat
   :target: https://landscape.io/github/Menelau/DESlib/master
   :alt: Code Health

.. image:: https://img.shields.io/badge/License-BSD%203--Clause-blue.svg
    :target: https://opensource.org/licenses/BSD-3-Clause


DESlib
========

DESlib is an easy to use ensemble learning library focusing the implementation of the state-of-the-art techniques for dynamic classifier and ensemble selection.
The library is is based on scikit-learn_, using the same method signature: **fit**, **predict**, **predict_proba** and **score**.
All dynamic selection techniques were implemented according to the definitions from [1]_.

Dynamic Selection:
-------------------

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


Dependencies:
-------------

DESlib is tested to work with Python 3.5, and 3.6. The dependency requirements are:

* scipy(>=0.13.3)
* numpy(>=1.10.4)
* scikit-learn(>=0.19.0)

These dependencies are automaticatically installed using the pip commands above.

Description:
------------

The library is divided into four modules:

1. deslib.des: Implementation of DES techniques.
2. deslib.dcs: Implementation of DCS techniques.
3. deslib.static: Implementation of baseline ensemble methods.
4. deslib.util: A collection of aggregation functions and diversity measures for ensemble of classifiers.

* DES techniques currently available are:
    1. META-DES [7]_ [8]_ [15]_
    2. K-Nearest-Oracle-Eliminate (KNORA-E) [3]_
    3. K-Nearest-Oracle-Union (KNORA-U) [3]_
    4. Dynamic Ensemble Selection-Performance(DES-P) [12]_
    5. K-Nearest-Output Profiles (KNOP) [9]_
    6. Randomized Reference Classifier (DES-RRC) [10]_
    7. DES Kullback-Leibler Divergence (DES-KL) [12]_
    8. DES-Exponential [21]_
    9. DES-Logarithmic [11]_
    10. DES-Minimum Difference [21]_
    11. DES-Clustering [16]_
    12. DES-KNN [16]_

* DCS techniques currently available are:
    1. Modified Classifier Rank (Rank) [19]_
    2. Overall Locall Accuracy (OLA) [4]_
    3. Local Class Accuracy (LCA) [4]_
    4. Modified Local Accuracy (MLA) [23]_
    5. Multiple Classifier Behaviour (MCB) [5]_
    6. A Priori Selection (A Priori) [6]_
    7. A Posteriori Selection (A Posteriori) [6]_

* Baseline methods:
    1. Oracle [20]_
    2. Single Best [2]_
    3. Static Selection [2]_

Variations of each DES techniques are also provided by the library (e.g., different versions of the META-DES framework).
Moreover, each DES technique can be used as Dynamic Selection, Dynamic Weighting or a Hybrid version which includes dynamic selection + weighting.
The library also offers the implementation of methods which can be applied to DS techniques in order to improve the classification
accuracy such as the Dynamic Frienemy Pruning (DFP) [13]_.

Examples:
---------

Here we show an example using the KNORA-E method using random forest as a pool of classifiers:

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

References:
-----------

.. [1] : R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,” Information Fusion, vol. 41, pp. 195 – 216, 2018.

.. [2] : A. S. Britto, R. Sabourin, L. E. S. de Oliveira, Dynamic selection of classifiers - A comprehensive review, Pattern Recognition 47 (11) (2014) 3665–3680.

.. [3] : A. H. R. Ko, R. Sabourin, u. S. Britto, Jr., From dynamic classifier selection to dynamic ensemble selection, Pattern Recognition 41 (2008) 1735–1748.

.. [4] : K. Woods, W. P. Kegelmeyer, Jr., K. Bowyer, Combination of multiple classifiers using local accuracy estimates, IEEE Transactions on Pattern Analysis Machine Intelligence 19 (1997) 405–410.

.. [5] : G. Giacinto, F. Roli, Dynamic classifier selection based on multiple classifier behaviour, Pattern Recognition 34 (2001) 1879–1881.

.. [6] : L. Didaci, G. Giacinto, F. Roli, G. L. Marcialis, A study on the performances of dynamic classifier selection based on local accuracy estimation, Pattern Recognition 38 (11) (2005) 2188–2191.

.. [7] : R. M. O. Cruz, R. Sabourin, G. D. C. Cavalcanti, T. I. Ren, META-DES: A dynamic ensemble selection framework using meta-learning, Pattern Recognition 48 (5) (2015) 1925–1935.

.. [8] : Cruz, R.M., Sabourin, R. and Cavalcanti, G.D., 2015, July. META-DES. H: a dynamic ensemble selection technique using meta-learning and a dynamic weighting approach. In Neural Networks (IJCNN), 2015 International Joint Conference on (pp. 1-8)

.. [9] : P. R. Cavalin, R. Sabourin, C. Y. Suen, Dynamic selection approaches for multiple classifier systems, Neural Computing and Applications 22 (3-4) (2013) 673–688.

.. [10] : T.Woloszynski, M. Kurzynski, A probabilistic model of classifier competence for dynamic ensemble selection, Pattern Recognition 44 (2011) 2656–2668.

.. [11] : T.Woloszynski, M. Kurzynski, A measure of competence based on randomized reference classifier for dynamic ensemble selection, in: International Conference on Pattern Recognition (ICPR), 2010, pp. 4194–4197.

.. [12] : T. Woloszynski, M. Kurzynski, P. Podsiadlo, G. W. Stachowiak, A measure of competence based on random classification for dynamic ensemble selection, Information Fusion 13 (3) (2012) 207–213.

.. [13] : Oliveira, D.V.R., Cavalcanti, G.D.C. and Sabourin, R., Online Pruning of Base Classifiers for Dynamic Ensemble Selection, Pattern Recognition, vol. 72, December 2017, pp 44-58.

.. [14] : Cruz RM, Zakane HH, Sabourin R, Cavalcanti GD. Dynamic Ensemble Selection VS K-NN: why and when Dynamic Selection obtains higher classification performance?.

.. [15] : R. M. O. Cruz, R. Sabourin, G. D. C. Cavalcanti, META-DES.Oracle: Meta-learning and feature selection for dynamic ensemble selection, Information Fusion 38 (2017) 84–103.Nov 30;38:84-103.

.. [16] : R. G. F. Soares, A. Santana, A. M. P. Canuto, M. C. P. de Souto, Using accuracy and diversity to select classifiers to build ensembles, Proceedings of the International Joint Conference on Neural Networks (2006) 1310–1316.

.. [17] : L. I. Kuncheva, Combining Pattern Classifiers: Methods and Algorithms, Wiley-Interscience, 2004.

.. [18] : Shipp, Catherine A., and Ludmila I. Kuncheva. "Relationships between combination methods and measures of diversity in combining classifiers." Information fusion 3.2 (2002): 135-148.

.. [19] : M. Sabourin, A. Mitiche, D. Thomas, G. Nagy, Classifier combination for handprinted digit recognition, International Conference on Document Analysis and Recognition (1993) 163–166.

.. [20] : L. I. Kuncheva, A theoretical study on six classifier fusion strategies, IEEE Transactions on Pattern Analysis and Machine Intelligence 24 (2) (2002) 281–286.

.. [21] : B. Antosik, M. Kurzynski, New measures of classifier competence – heuristics and application to the design of multiple classifier systems., in: Computer recognition systems 4., 2011, pp. 197–206.

.. [22] : Smith, M.R., Martinez, T. and Giraud-Carrier, C., 2014. An instance level analysis of data complexity. Machine learning, 95(2), pp.225-256

.. [23] : P. C. Smits, Multiple classifier systems for supervised remote sensing image classification based on dynamic classifier selection, IEEE Transactions on Geoscience and Remote Sensing 40 (4) (2002) 801–813.

.. _scikit-learn: http://scikit-learn.org/stable/

.. _numpy: http://www.numpy.org/

.. _scipy: https://www.scipy.org/
