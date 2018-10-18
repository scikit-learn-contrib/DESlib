.. _known_issues:

Known Issues
============

The estimators in this library are not compatible with scikit-learn's GridSearch, and other CV methods. That is, the following is not supported:

.. code-block:: python

    from deslib.des.knora_e import KNORAE
    from sklearn.model_selection import GridSearchCV

    # (...) initialize a pool of classifiers
    kne = KNORAE(pool_classifiers)

    # Do a grid search on KNORAE's "k" parameter
    params = {'k': [1, 3, 5, 7]}

    grid = GridSearchCV(kne, params)
    grid.fit(X_dsel, y_dsel)  # Raises an error

This is due to a limitation of a scikit-learn method (sklearn.base.clone), under discussion in this issue_

.. _issue: https://github.com/scikit-learn/scikit-learn/issues/8370