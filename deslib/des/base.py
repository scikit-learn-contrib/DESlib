from abc import ABCMeta

import numpy as np

from deslib.base import BaseDS
from deslib.util.aggregation import (aggregate_proba_ensemble_weighted,
                                     sum_votes_per_class,
                                     get_weighted_votes)


class BaseDES(BaseDS):
    """Base class for a Dynamic Ensemble Selection (DES).

    All dynamic ensemble selection techniques should inherit from this class.

    Warning: This class should not be instantiated directly, use
    derived classes instead.

    """
    __metaclass__ = ABCMeta

    def __init__(self, pool_classifiers=None, k=7, DFP=False, with_IH=False,
                 safe_k=None, IH_rate=0.30, mode='selection', voting='hard',
                 needs_proba=False, random_state=None,
                 knn_classifier='knn', knn_metric='minkowski', knne=False,
                 DSEL_perc=0.5, n_jobs=-1):

        super(BaseDES, self).__init__(pool_classifiers=pool_classifiers,
                                      k=k,
                                      DFP=DFP,
                                      with_IH=with_IH,
                                      safe_k=safe_k,
                                      IH_rate=IH_rate,
                                      needs_proba=needs_proba,
                                      random_state=random_state,
                                      knn_classifier=knn_classifier,
                                      knn_metric=knn_metric,
                                      knne=knne,
                                      DSEL_perc=DSEL_perc, n_jobs=n_jobs)
        self.mode = mode
        self.voting = voting

    def classify_with_ds(self, predictions, probabilities=None,
                         competence_region=None, distances=None,
                         DFP_mask=None):
        """Predicts the label of the corresponding query sample.

        If self.mode == "selection", the selected ensemble is combined using
        the majority voting rule

        If self.mode == "weighting", all base classifiers are used for
        classification, however their influence in the final decision are
        weighted according to their estimated competence level. The weighted
        majority voting scheme is used to combine the decisions of the
        base classifiers.

        If self.mode == "hybrid",  A hybrid Dynamic selection and weighting
        approach is used. First an ensemble with the competent base classifiers
        are selected. Then, their decisions are aggregated using the weighted
        majority voting rule according to its competence level estimates.

        Parameters
        ----------
        predictions : array of shape (n_samples, n_classifiers)
                      Predictions of the base classifier for all test examples.

        probabilities : array of shape (n_samples, n_classifiers, n_classes)
            Probabilities estimates of each base classifier for all test
            examples. (For methods that always require probabilities from
            the base classifiers).

        competence_region : array of shape (n_samples, n_neighbors)
            Indices of the k nearest neighbors according for each test sample.

        distances : array of shape (n_samples, n_neighbors)
                        Distances from the k nearest neighbors to the query

        DFP_mask : array of shape (n_samples, n_classifiers)
            Mask containing 1 for the selected base classifier and 0 otherwise.

        Returns
        -------
        predicted_label : array of shape (n_samples)
                          Predicted class label for each test example.
        """
        probas = self.predict_proba_with_ds(predictions, probabilities,
                                            competence_region, distances,
                                            DFP_mask)
        return probas.argmax(axis=1)

    def predict_proba_with_ds(self, predictions, probabilities=None,
                              competence_region=None, distances=None,
                              DFP_mask=None):
        """Predicts the posterior probabilities of the corresponding query.

        If self.mode == "selection", the selected ensemble is used to estimate
        the probabilities. The average rule is used
        to give probabilities estimates.

        If self.mode == "weighting", all base classifiers are used for
        estimating the probabilities, however their influence in the final
        decision are weighted according to their estimated competence level.
        A weighted average method is used to give the probabilities estimates.

        If self.mode == "Hybrid",  A hybrid Dynamic selection and weighting
        approach is used. First an ensemble with the competent base classifiers
        are selected. Then, their decisions are aggregated using a weighted
        average rule to give the probabilities estimates.

        Parameters
        ----------
        predictions : array of shape (n_samples, n_classifiers)
            Predictions of the base classifier for all test examples.

        probabilities : array of shape (n_samples, n_classifiers, n_classes)
            Probabilities estimates of each base classifier for all samples.

        competence_region : array of shape (n_samples, n_neighbors)
            Indices of the k nearest neighbors.
        distances : array of shape (n_samples, n_neighbors)
            Distances from the k nearest neighbors to the query

        DFP_mask : array of shape (n_samples, n_classifiers)
            Mask containing 1 for the selected base classifier and 0 otherwise.

        Returns
        -------
        predicted_proba : array = [n_samples, n_classes]
                          The probability estimates for all test examples.
        """
        if self.needs_proba:
            competences = self.estimate_competence_from_proba(
                neighbors=competence_region,
                distances=distances,
                probabilities=probabilities)
        else:
            competences = self.estimate_competence(
                competence_region=competence_region,
                distances=distances,
                predictions=predictions)
        if self.DFP:
            # FIRE-DES pruning.
            competences = competences * DFP_mask

        if self.mode == "selection":
            predicted_proba = self._dynamic_selection(competences,
                                                      predictions,
                                                      probabilities)
        elif self.mode == "weighting":
            predicted_proba = self._dynamic_weighting(competences, predictions,
                                                      probabilities)
        else:
            predicted_proba = self._hybrid(competences, predictions,
                                           probabilities)

        return predicted_proba

    def _dynamic_selection(self, competences, predictions, probabilities):
        """ Combine models using dynamic ensemble selection. """
        selected_classifiers = self.select(competences)
        if self.voting == 'hard':
            votes = np.ma.MaskedArray(predictions, ~selected_classifiers)
            votes = sum_votes_per_class(votes, self.n_classes_)
            predicted_proba = votes / votes.sum(axis=1)[:, None]
        else:
            masked_proba = self._mask_proba(probabilities,
                                            selected_classifiers)
            predicted_proba = np.mean(masked_proba, axis=1)
        return predicted_proba

    def _dynamic_weighting(self, competences, predictions, probabilities):
        """ Combine models using dynamic weighting. """
        if self.voting == 'hard':
            w_votes, _ = get_weighted_votes(predictions,
                                            competences,
                                            np.arange(self.n_classes_))
            predicted_proba = w_votes / w_votes.sum(axis=1)[:, None]
        else:
            predicted_proba = aggregate_proba_ensemble_weighted(
                probabilities, competences)
        return predicted_proba

    def _hybrid(self, competences, predictions, probabilities):
        """ Combine models using a hybrid dynamic selection + weighting. """
        selected_classifiers = self.select(competences)
        if self.voting == 'hard':
            votes = np.ma.MaskedArray(predictions, ~selected_classifiers)
            w_votes, _ = get_weighted_votes(votes, competences,
                                            np.arange(self.n_classes_))
            predicted_proba = w_votes / w_votes.sum(axis=1)[:, None]
        else:
            masked_proba = self._mask_proba(probabilities,
                                            selected_classifiers)
            predicted_proba = aggregate_proba_ensemble_weighted(
                masked_proba, competences)
        return predicted_proba

    @staticmethod
    def _mask_proba(probabilities, selected_classifiers):
        # Broadcast the selected classifiers mask
        # to cover the last axis (n_classes):
        selected_classifiers = np.expand_dims(selected_classifiers,
                                              axis=2)
        selected_classifiers = np.broadcast_to(selected_classifiers,
                                               probabilities.shape)
        masked_proba = np.ma.MaskedArray(probabilities,
                                         ~selected_classifiers)
        return masked_proba

    def _validate_parameters(self):
        super(BaseDES, self)._validate_parameters()

        if not isinstance(self.mode, str):
            raise TypeError(
                'Parameter "mode" should be a string.'
                ' Currently "mode" = {}' .format(type(self.mode)))

        if self.mode not in ['selection', 'hybrid', 'weighting']:
            raise ValueError(
                'Invalid value for parameter "mode".'
                ' "mode" should be one of these options '
                '{selection, hybrid, weighting}')

        if self.voting not in ['soft', 'hard']:
            raise ValueError('Invalid value for parameter "voting".'
                             ' "voting" should be one of these options '
                             '{selection, hybrid, weighting}')
        if self.voting == 'soft':
            self._check_predict_proba()
