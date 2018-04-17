.. _api-reference:

######################
API Reference
######################

This is the full API documentation of the `DESlib`. Currently the library is divided into four modules:

Dynamic Classifier Selection (DCS)
-----------------------------------

This module contains the implementation of techniques in which only the base
classifier that attained the highest competence level is selected for the classification of the query.

.. automodule:: deslib.dcs

.. toctree::
    :maxdepth: 3

    modules/dcs/base
    modules/dcs/a_posteriori
    modules/dcs/a_priori
    modules/dcs/lca
    modules/dcs/mcb
    modules/dcs/mla
    modules/dcs/ola
    modules/dcs/rank

Dynamic Ensemble Selection (DES)
-----------------------------------

Dynamic ensemble selection strategies refer to techniques that select an ensemble of classifier rather than a single one.
All base classifiers that attain a minimum competence level are selected to compose the ensemble of classifiers.

.. automodule:: deslib.des

.. toctree::
    :maxdepth: 3

    modules/des/base
    modules/des/meta_des
    modules/des/des_clustering
    modules/des/des_p
    modules/des/ds_knn
    modules/des/knop
    modules/des/knora_e
    modules/des/knora_u
    modules/des/probabilistic

Static ensembles
-----------------------------------

This module provides the implementation of static ensemble techniques that are usually used as a baseline for the
comparison of DS methods: Single Best (SB), Static Selection (SS) and Oracle.


.. automodule:: deslib.static

.. toctree::
    :maxdepth: 3

    modules/static/oracle
    modules/static/single_best
    modules/static/static_selection

Utils
-----------------------------------
Utility functions for ensemble methods such as diversity and aggregation methods.

.. automodule:: deslib.util

.. toctree::
    :maxdepth: 3

    modules/util/diversity
    modules/util/aggregation
    modules/util/prob_functions
    modules/util/instance_hardness
