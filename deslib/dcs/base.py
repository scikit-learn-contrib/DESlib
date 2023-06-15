from abc import ABCMeta
from copy import copy

import numpy as np
from sklearn.utils.validation import check_random_state

from deslib.base import BaseDS
from deslib.util.aggregation import majority_voting_rule


class BaseDCS(BaseDS):
    """Base class for a Dynamic Classifier Selection (dcs) method.
    All dynamic classifier selection classes should inherit from this class.

    Warning: This class should not be used directly, use derived classes
    instead.

    """
    __metaclass__ = ABCMeta

    def __init__(self, pool_classifiers=None, k=7, DFP=False, safe_k=None,
                 with_IH=False, IH_rate=0.30, selection_method='best',
                 diff_thresh=0.1, random_state=None, knn_classifier='knn',
                 knn_metric='minkowski', DSEL_perc=0.5,
                 knne=False, n_jobs=-1):

        super(BaseDCS, self).__init__(pool_classifiers=pool_classifiers, k=k,
                                      DFP=DFP, with_IH=with_IH, safe_k=safe_k,
                                      IH_rate=IH_rate,
                                      random_state=random_state,
                                      knn_classifier=knn_classifier,
                                      knn_metric=knn_metric,
                                      DSEL_perc=DSEL_perc,
                                      knne=knne, n_jobs=n_jobs,
                                      voting='single')

        self.selection_method = selection_method
        self.diff_thresh = diff_thresh

    def estimate_competence(self, competence_region, distances=None,
                            predictions=None):
        """Estimate the competence of each base classifier for the
        classification of the query sample.

        Parameters
        ----------
        competence_region : array of shape (n_samples, n_neighbors)
            Indices of the k nearest neighbors.

        distances : array of shape (n_samples, n_neighbors)
            Distances from the k nearest neighbors to the query.

        predictions : array of shape (n_samples, n_classifiers)
            Predictions of the base classifiers for the test examples.

        Returns
        -------
        competences : array of shape (n_samples, n_classifiers)
            Competence level estimated for each base classifier and test
            example.
        """
        pass

    def select(self, competences):
        """Select the most competent classifier for the classification of the
        query sample given the competence level estimates. Four selection
        schemes are available.

        Best : The base classifier with the highest competence level is
        selected. In cases where more than one base classifier achieves the
        same competence level, the one with the lowest index is selected. This
        method is the standard for the LCA, OLA, MLA techniques.

        Diff : Select the base classifier that is significantly better than the
        others in the pool (when the difference between its competence level
        and the competence level of the other base classifiers is higher than a
        predefined threshold). If no base classifier is significantly better,
        the base classifier is selected randomly among the member with
        equivalent competence level.

        Random : Selects a random base classifier among all base classifiers
        that achieved the same competence level.

        ALL : all base classifiers with the max competence level estimates are
        selected (note that in this case the
        DCS technique becomes a DES technique).

        Parameters
        ----------
        competences : array of shape (n_samples, n_classifiers)
            Competence level estimated for each base classifier and test
            example.

        Returns
        -------
        selected_classifiers : array of shape [n_samples]
            Indices of the selected base classifier for each sample. If the
            selection_method is set to 'all', a boolean matrix is returned,
            containing True for the selected base classifiers, otherwise false.

        """
        if competences.ndim < 2:
            competences = competences.reshape(1, -1)

        selected_classifiers = []
        best_index = np.argmax(competences, axis=1)

        if self.selection_method == 'best':
            # Select the classifier with highest competence level
            selected_classifiers = best_index

        elif self.selection_method == 'diff':
            rng = check_random_state(copy(self.random_state))
            best_competence = competences[
                np.arange(competences.shape[0]), best_index]
            # best_competence = np.max(competences)
            diff = best_competence.reshape(-1, 1) - competences
            # TODO: Improve this part of the code
            selected_classifiers = np.zeros(diff.shape[0], dtype=int)
            for row in range(diff.shape[0]):
                diff_list = list(diff[row, :])
                indices = [idx for idx, _ in enumerate(diff_list) if
                           diff_list[idx] < self.diff_thresh]

                if len(indices) == 0:
                    indices = range(self.n_classifiers_)

                selected_classifiers[row] = rng.choice(indices)

        elif self.selection_method == 'random':
            # TODO: Improve this part of the code
            rng = check_random_state(copy(self.random_state))
            selected_classifiers = np.zeros(competences.shape[0], dtype=int)
            best_competence = competences[
                np.arange(competences.shape[0]), best_index]
            for row in range(competences.shape[0]):
                competence_list = list(competences[row, :])

                # Select a random classifier among all with same competence
                # level
                indices = [idx for idx, _ in enumerate(competence_list) if
                           competence_list[idx] == best_competence[row]]

                selected_classifiers[row] = rng.choice(indices)

        elif self.selection_method == 'all':
            # select all base classifiers with max competence estimates.
            max_value = np.max(competences, axis=1)
            selected_classifiers = (
                        competences == max_value.reshape(competences.shape[0],
                                                         -1))

        return selected_classifiers

    def classify_with_ds(self, predictions, probabilities=None,
                         neighbors=None, distances=None, DFP_mask=None):
        """Predicts the class label of the corresponding query sample.

        If self.selection_method == "all", the majority voting scheme is used
        to aggregate the predictions of all classifiers with the max competence
        level estimates for each test examples.

        Parameters
        ----------
        predictions : array of shape (n_samples, n_classifiers)
            Predictions of the base classifiers for all test examples

        probabilities : array of shape (n_samples, n_classifiers, n_classes)
            Probabilities estimates of each base classifier for all test
            examples (For methods that always require probabilities from the
            base classifiers)

        neighbors : array of shape (n_samples, n_neighbors)
            Indices of the k nearest neighbors.
        distances : array of shape (n_samples, n_neighbors)
            Distances from the k nearest neighbors to the query

        DFP_mask : array of shape (n_samples, n_classifiers)
            Mask containing 1 for the selected base classifier and 0 otherwise.

        Returns
        -------
        predicted_label : array of shape (n_samples)
            The predicted label for each query
        """
        if predictions.ndim < 2:
            predictions = predictions.reshape(1, -1)
        competences = self.estimate_competence(neighbors,
                                               distances=distances,
                                               predictions=predictions)

        if self.DFP:
            competences = competences * DFP_mask

        if self.selection_method != 'all':
            # only one classifier is selected
            clf_index = self.select(competences)
            predicted_label = predictions[
                np.arange(predictions.shape[0]), clf_index]
        else:
            # Selected ensemble of classifiers is combined using Majority
            # Voting
            indices = self.select(competences)
            votes = np.ma.MaskedArray(predictions, ~indices)
            predicted_label = majority_voting_rule(votes)

        return predicted_label

    def predict_proba_with_ds(self, predictions, probabilities,
                              neighbors=None, distances=None, DFP_mask=None):
        """Predicts the posterior probabilities of the corresponding query
        sample.

        If self.selection_method == "all", get the probability estimates of the
        selected ensemble. Otherwise, the technique gets the probability
        estimates from the selected base classifier

        Parameters
        ----------
        predictions : array of shape (n_samples, n_classifiers)
            Predictions of the base classifiers for all test examples

        probabilities : array of shape (n_samples, n_classifiers, n_classes)
            The predictions of each base classifier for all samples (For
            methods that always require probabilities from the base
            classifiers).

        neighbors : array of shape (n_samples, n_neighbors)
            Indices of the k nearest neighbors.
        distances : array of shape (n_samples, n_neighbors)
            Distances from the k nearest neighbors to the query

        DFP_mask : array of shape (n_samples, n_classifiers)
           Mask containing 1 for the selected base classifier and 0 otherwise.

        Returns
        -------
        predicted_proba: array of shape (n_samples, n_classes)
            Posterior probabilities estimates for each test example.
        """
        competences = self.estimate_competence(neighbors,
                                               distances=distances,
                                               predictions=predictions)

        if self.DFP:
            competences = competences * DFP_mask

        if self.selection_method != 'all':
            # only one classifier is selected
            clf_index = self.select(competences)
            predicted_proba = probabilities[
                np.arange(probabilities.shape[0]), clf_index]
        else:
            # Selected ensemble of classifiers is combined using average
            # probability
            selected_classifiers = self.select(competences)

            # Broadcast the selected classifiers mask (to cover the last axis
            # (nClasses):
            selected_classifiers = np.expand_dims(selected_classifiers, axis=2)
            selected_classifiers = np.broadcast_to(selected_classifiers,
                                                   probabilities.shape)
            masked_proba = np.ma.MaskedArray(probabilities,
                                             ~selected_classifiers)

            predicted_proba = np.mean(masked_proba, axis=1)

        return predicted_proba

    def _validate_parameters(self):

        super(BaseDCS, self)._validate_parameters()

        if not isinstance(self.selection_method, str):
            raise TypeError(
                'The parameter selection_method should be a string.'
                ' selection_method = ', type(self.selection_method))

        if self.selection_method not in ['best', 'all', 'random', 'diff']:
            raise ValueError(
                'Invalid value for parameter "selection_method." '
                'The possible values are: '
                '"best", "all", "random", "diff"')

        if not isinstance(self.diff_thresh, float):
            raise TypeError(
                'The parameter diff_thresh should be a float. diff_thresh = ',
                self.diff_thresh)

        if self.diff_thresh >= 0.5 or self.diff_thresh < 0.0 or np.isnan(
                self.diff_thresh):
            raise ValueError(
                'diff_thresh should be lower than 0.5. diff_thresh = ',
                self.diff_thresh)
