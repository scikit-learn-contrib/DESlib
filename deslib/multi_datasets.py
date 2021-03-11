# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import copy
import numpy as np
from scipy.stats import mode
from sklearn.utils.validation import (check_is_fitted, check_array)

from deslib.base import BaseDS
from deslib.dcs.base import BaseDCS
from deslib.des.base import BaseDES
from deslib.static.oracle import Oracle
from deslib.static.stacked import StackedClassifier
from deslib.util.aggregation import (weighted_majority_voting_rule,
                                     majority_voting_rule,
                                     aggregate_proba_ensemble_weighted)
from deslib.util.instance_hardness import hardness_region_competence


# Créer à partir de KNORA-U
class MultiDatasets(BaseDS):
    def __init__(self, ds_classifier, pool_classifiers):
        """
        Parameters
        ----------
        ds_classifier : classifier from the library
            The DS model serves as a template for all the datasets.

        pool_classifiers : array of shape (n_datasets, n_classifiers)
            Classifiers of each dataset.
        """
        super(MultiDatasets, self).__init__(pool_classifiers)
        self.ds_classifier = ds_classifier

    def fit(self, X, y):
        """Prepare the DS models by setting the KNN algorithm and
        pre-processing the information required to apply the DS
        methods

        Parameters
        ----------
        X : array of shape (n_datasets, n_samples, n_features)
            The input data.

        y : array of shape (n_datasets, n_samples)
            class labels of each example in X.
        """
        n_datasets = len(X)
        self.ds_classifiers = []

        for i in range(n_datasets):
            ds_classifier = copy.deepcopy(self.ds_classifier)
            ds_classifier.pool_classifiers = self.pool_classifiers[i]
            ds_classifier.fit(X[i], y[i])
            self.ds_classifiers.append(ds_classifier)

        one_classifier = self.ds_classifiers[0]
        self.ds_classifier.n_classes_ = one_classifier.n_classes_
        self._setup_label_encoder(y[0])

        return self

    def predict(self, X, y=None):
        """
        Parameters
        ----------
        X : array of shape (n_datasets, n_samples, n_features)
            The input data.

        y : array of shape (n_datasets, n_samples)
            class labels of each example in X.
            It's added as a parameter so Oracle can be used.
        """
        # Oracle is not a DS model, so there is no need to execute
        # the code below.
        if issubclass(type(self.ds_classifier), Oracle):
            return self._predict_oracle(X, y)
        elif issubclass(type(self.ds_classifier), StackedClassifier):
            return self._predict_stacked(X)
        else:
            return self._predict_ds(X)

    def _predict_ds(self, X):
        merged_base_probabilities = []
        merged_base_predictions = []
        n_datasets = len(X)

        for i in range(n_datasets):
            base_probabilities, base_predictions = \
                self._get_base_proba_and_pred(self.ds_classifiers[i], X[i])
            merged_base_probabilities.append(base_probabilities)
            merged_base_predictions.append(base_predictions)

        if merged_base_probabilities[0] is not None:
            merged_base_probabilities = np.concatenate(
                merged_base_probabilities, axis=1)
        else:
            merged_base_probabilities = None
        merged_base_predictions = np.concatenate(
                merged_base_predictions, axis=1)

        n_samples = len(X[0])
        predicted_labels = np.empty(n_samples, dtype=np.intp)

        all_agree_vector = BaseDS._all_classifier_agree(merged_base_predictions)
        ind_all_agree = np.where(all_agree_vector)[0]   

        # Since the predictions are always the same, get the predictions of the
        # first base classifier.
        if ind_all_agree.size:
            predicted_labels[ind_all_agree] = merged_base_predictions[
                ind_all_agree, 0]

        # For the samples with disagreement, perform the dynamic selection
        # steps. First step is to collect the samples with disagreement
        # between base classifiers
        ind_disagreement = np.where(~all_agree_vector)[0]
        if ind_disagreement.size:
            merged_left_base_predictions = []
            merged_competences = []
            for i in range(n_datasets):
                ds_classifier = self.ds_classifiers[i]
                X_DS = X[i][ind_disagreement, :]
                base_probabilities, base_predictions = \
                    self._get_base_proba_and_pred(ds_classifier, X[i])

                # If the method is based on clustering and does not use IH there
                # is no need to compute the Neighbors
                if hasattr(self.ds_classifier, "clustering_") \
                        and not ds_classifier.with_IH:
                    distances = neighbors = None
                else:
                    # Then, we estimate the nearest neighbors for all samples
                    # that we need to call DS routines
                    distances, neighbors = \
                        ds_classifier._get_region_competence(X_DS)

                if ds_classifier.with_IH:
                    raise ValueError("TODO: traiter avec tous les inputs.")
                    ind_ds_classifier, predicted_labels, neighbors, \
                    distances = \
                        self._calculate_hardness_level(ds_classifier,
                                                       ind_disagreement,
                                                       predicted_labels,
                                                       neighbors,
                                                       distances)
                else:
                    # IH was not considered. So all samples with disagreement
                    # are passed down to the DS algorithm
                    ind_ds_classifier = np.arange(ind_disagreement.size)

                # At this stage the samples which all base classifiers agrees or
                # that are associated with low hardness were already classified.
                # The remaining samples are now passed down to the DS techniques
                # for classification.

                #  First check whether there are still samples to be classified.
                if ind_ds_classifier.size:
                    # IF the DFP pruning is considered, calculate the DFP mask
                    # for all samples in X
                    DFP_mask = self._get_DFP_mask(
                        ds_classifier, ind_ds_classifier, neighbors)

                # Get the real indices_ of the samples that will be classified
                # using a DS algorithm.
                ind_ds_original_matrix = ind_disagreement[ind_ds_classifier]
                self.stats.disagree_ind = ind_ds_original_matrix

                if ds_classifier.needs_proba:
                    selected_probabilities = base_probabilities[
                        ind_ds_original_matrix]
                else:
                    selected_probabilities = None

                competences = self._get_competences(
                    ds_classifier,
                    X_DS[ind_ds_classifier],
                    base_predictions[ind_ds_original_matrix],
                    selected_probabilities,
                    neighbors=neighbors,
                    distances=distances,
                    DFP_mask=DFP_mask)

                merged_competences.append(competences)
                merged_left_base_predictions.append(base_predictions[
                                                    ind_ds_original_matrix])

            merged_left_base_predictions = np.concatenate(
                merged_left_base_predictions, axis=1)
            merged_competences = np.concatenate(merged_competences, axis=1)
            self.stats.competences = merged_competences

            if issubclass(type(self.ds_classifier), BaseDCS):
                pred_ds = self._get_dcs_predicted_label(self.ds_classifier,
                    merged_left_base_predictions, merged_competences)
            elif issubclass(type(self.ds_classifier), BaseDES):
                pred_ds = self._get_des_predicted_label(self.ds_classifier,
                    merged_left_base_predictions, merged_competences)

            predicted_labels[ind_ds_original_matrix] = pred_ds

        self.stats.bases_labels = merged_base_predictions
        self.stats.agree_ind = ind_all_agree
        self.stats.predicted_labels = predicted_labels

        return self.classes_.take(predicted_labels)

    def _predict_oracle(self, X, y):
        n_datasets = len(X)
        predicted_labels = -np.ones(y.size, dtype=int)

        for sample_index in range(len(y)):
            predictions = []

            for i in range(n_datasets):
                classifier = self.ds_classifiers[i]
                X[i] = check_array(X[i])
                y = classifier.enc_.transform(y)
                x_sample = X[i][sample_index]
                y_sample = y[sample_index]

                for clf in classifier.pool_classifiers_:
                    predictions.append(clf.predict(x_sample.reshape(1, -1))[0])

            for p in predictions:
                # If one base classifier predicts the correct answer,
                # consider as a correct prediction
                if p == y_sample:
                    p = int(p)
                    predicted_labels[sample_index] = p
                    break
                predicted_labels[sample_index] = p

        return self.classes_.take(predicted_labels)

    def _predict_stacked(self, X):
        merged_base_preds = []
        n_datasets = len(X)

        for i in range(n_datasets):
            classifier = self.ds_classifiers[i]
            X[i] = check_array(X[i])
            check_is_fitted(classifier, "meta_classifier_")
            base_preds = classifier._predict_proba_base(X[i])
            X_meta = classifier._connect_input(X[i], base_preds)
            preds = classifier.meta_classifier_.predict_proba(X_meta)
            merged_base_preds.append(preds)

        merged_base_preds = np.sum(merged_base_preds,0)
        preds = np.argmax(merged_base_preds,axis=1)

        return self.classes_.take(preds)

    def predict_proba(self, X):
        raise ValueError("Méthode incomplète!")

    def _get_base_proba_and_pred(self, ds_classifier, X):
        """
        This code has been copied, but "self" has been replaced by
        "ds_classifier" because this function doesn't exist elsewhere.
        """
        self = ds_classifier
        # Check if the DS model was trained
        check_is_fitted(self,
                        ["DSEL_processed_", "DSEL_data_", "DSEL_target_"])

        # Check if X is a valid input
        X = check_array(X)
        self._check_num_features(X)

        if self.needs_proba:
            base_probabilities = self._predict_proba_base(X)
            base_predictions = base_probabilities.argmax(axis=2)
        else:
            base_probabilities = None
            base_predictions = self._predict_base(X)

        return base_probabilities, base_predictions

    def _calculate_hardness_level(self, ds_classifier, ind_disagreement,
            predicted_labels, neighbors, distances):
        """
        This code has been copied, but "self" has been replaced by
        "ds_classifier" because this function doesn't exist elsewhere.
        """
        self = ds_classifier

        # if IH is used, calculate the hardness level associated with
        # each sample
        hardness = hardness_region_competence(neighbors,
                                              self.DSEL_target_,
                                              self.safe_k)

        # Get the index associated with the easy and hard samples.
        # Samples with low hardness are passed down to the knn
        # classifier while samples with high hardness are passed down
        # to the DS methods. So, here we split the samples that are
        # passed to down to each stage by calculating their indices_.
        easy_samples_mask = hardness < self.IH_rate
        ind_knn_classifier = np.where(easy_samples_mask)[0]
        ind_ds_classifier = np.where(~easy_samples_mask)[0]

        if ind_knn_classifier.size:
            # all samples with low hardness should be classified by
            # the knn method here:
            # First get the class associated with each neighbor
            y_neighbors = self.DSEL_target_[
                neighbors[ind_knn_classifier, :self.safe_k]]

            # Accessing which samples in the original matrix are
            # associated with the low instance hardness indices_. This
            # is important since the low hardness indices
            # ind_knn_classifier was estimated based on a subset
            # of samples
            ind_knn_original_matrix = ind_disagreement[ind_knn_classifier]
            prediction_knn, _ = mode(y_neighbors, axis=1)
            predicted_labels[
                ind_knn_original_matrix] = prediction_knn.reshape(-1, )

            # Remove from the neighbors and distance matrices the
            # samples that were classified using the KNN
            neighbors = np.delete(neighbors, ind_knn_classifier,axis=0)
            distances = np.delete(distances, ind_knn_classifier,axis=0)

        return ind_ds_classifier, predicted_labels, neighbors, distances

    def _get_DFP_mask(self, ds_classifier, ind_ds_classifier, neighbors):
        """
        This code has been copied, but "self" has been replaced by
        "ds_classifier" because this function doesn't exist elsewhere.
        """
        self = ds_classifier

        if self.DFP:
            DFP_mask = frienemy_pruning_preprocessed(
                neighbors,
                self.DSEL_target_,
                self.DSEL_processed_)
        else:
            DFP_mask = np.ones(
                (ind_ds_classifier.size, self.n_classifiers_))

    def _get_competences(self, ds_classifier, query, predictions, 
                        probabilities=None, neighbors=None, distances=None,
                        DFP_mask=None):
        """
        This code has been copied, but "self" has been replaced by
        "ds_classifier" because this function doesn't exist elsewhere.
        """
        self = ds_classifier

        if query.ndim < 2:
            query = query.reshape(1, -1)

        if predictions.ndim < 2:
            predictions = predictions.reshape(1, -1)

        if query.shape[0] != predictions.shape[0]:
            raise ValueError(
                'The arrays query and predictions must have the same number'
                ' of samples. query.shape is {}'
                'and predictions.shape is {}'.format(query.shape,
                                                     predictions.shape))

        if self.needs_proba:
            competences = self.estimate_competence_from_proba(
                query,
                neighbors=neighbors,
                distances=distances,
                probabilities=probabilities)
        else:
            competences = self.estimate_competence(
                query,
                neighbors=neighbors,
                distances=distances,
                predictions=predictions)

        if self.DFP: competences = competences * DFP_mask

        return competences

    def _get_dcs_predicted_label(self, ds_classifier, predictions, competences):
        """
        This code has been copied, but "self" has been replaced by
        "ds_classifier" because this function doesn't exist elsewhere.
        """
        self = ds_classifier

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

    def _get_des_predicted_label(self, ds_classifier, predictions, competences):
        """
        This code has been copied, but "self" has been replaced by
        "ds_classifier" because this function doesn't exist elsewhere.
        """
        self = ds_classifier

        if self.mode == "selection":
            # The selected_classifiers matrix is used as a mask to remove
            # the predictions of certain base classifiers.
            selected_classifiers = self.select(competences)
            votes = np.ma.MaskedArray(predictions, ~selected_classifiers)
            predicted_label = majority_voting_rule(votes)
        elif self.mode == "weighting":
            votes = np.atleast_2d(predictions)
            predicted_label = weighted_majority_voting_rule(votes, competences,
                np.arange(self.n_classes_))
        else:
            selected_classifiers = self.select(competences)
            votes = np.ma.MaskedArray(predictions, ~selected_classifiers)
            predicted_label = weighted_majority_voting_rule(votes, competences,
                np.arange(self.n_classes_))

        return predicted_label
