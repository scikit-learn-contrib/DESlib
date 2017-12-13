# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier


def generate_pool(X, y, classifier_model,
                  num_classifiers=10, method='bagging', n_jobs=-1):
    method.lower()
    assert method in ['bagging', 'boosting', 'random']
    if method == 'bagging':
        pool_classifiers = BaggingClassifier(
            base_estimator=classifier_model, n_estimators=num_classifiers, n_jobs=n_jobs)

    elif method == 'boosting':

        pool_classifiers = AdaBoostClassifier(
            base_estimator=classifier_model, n_estimators=num_classifiers, n_jobs=n_jobs)

    else:
        pool_classifiers = RandomForestClassifier(n_estimators=num_classifiers, n_jobs=n_jobs)

    pool_classifiers.fit(X, y)

    return pool_classifiers

