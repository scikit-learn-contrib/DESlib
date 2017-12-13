# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause
import collections


def majority_voting(ensemble, query):
    """Performs a majority voting combination scheme between the base classifiers
    specified in the vector indices. Returns the label of the query sample as the
    most voted class.

    Parameters
    ----------
    ensemble : The ensemble of classifiers to be used in the aggregation scheme

    query : Sample to be classified

    Returns
    -------
    predicted_label : The label of the query sample, predicted by the majority voting rule
    """
    votes = [clf.predict(query)[0] for clf in ensemble]
    counter = collections.Counter(votes)
    predicted_label = counter.most_common()[0][0]

    return predicted_label


def weighted_majority_voting(ensemble, weights, query):
        """Performs a majority voting combination scheme between the base classifiers
        specified in the vector indices. Returns the label of the query sample as the
        most voted class.

        Parameters
        ----------
        indices : index of the base classifier to be used in the combination scheme

        weights : the weights associated to each classifier for the combination scheme

        query : Sample to be classified

        Returns
        -------
        predicted_label : The label of the query sample, predicted by the majority voting rule
        """
        w_votes = np.zeros(self.n_classes)
        for idx, clf in enumerate(ensemble):
            w_votes[clf.predict(query)[0]] += weights[idx]

        predicted_label = np.argmax(w_votes)
        return predicted_label


def averaging(ensemble, query):

    for clf in ensemble:
        clf.predict_proba(query)


    predicted_label = np.argmax(proba)

    return predicted_label


def product(ensemble, weights, query):
    predicted_label = None

    return predicted_label


def maximum(ensemble, weights, query):
    predicted_label = None

    return predicted_label

def median(ensemble, weights, query):
    predicted_label = None

    return predicted_label