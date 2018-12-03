# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause


from abc import abstractmethod, ABCMeta

import math
import numpy as np
import functools
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaseEnsemble, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.validation import (check_X_y, check_is_fitted, check_array,
                                      check_random_state)
from deslib.util import faiss_knn_wrapper

from deslib.util.instance_hardness import hardness_region_competence
import warnings


class BaseDS(BaseEstimator, ClassifierMixin):
    """Base class for a dynamic classifier selection (dcs) and
       dynamic ensemble selection (des) methods.

    All dcs and des techniques should inherit from this class.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, pool_classifiers=None, k=7, DFP=False, with_IH=False,
                 safe_k=None, IH_rate=0.30, needs_proba=False,
                 random_state=None, knn_classifier='knn', DSEL_perc=0.5):

        self.pool_classifiers = pool_classifiers
        self.k = k
        self.DFP = DFP
        self.with_IH = with_IH
        self.safe_k = safe_k
        self.IH_rate = IH_rate
        self.needs_proba = needs_proba
        self.random_state = random_state
        self.knn_classifier = knn_classifier
        self.DSEL_perc = DSEL_perc

        # Check optional dependency
        if knn_classifier == 'faiss' and not faiss_knn_wrapper.is_available():
            raise ImportError(
                'Using knn_classifier="faiss" requires that the FAISS library '
                'be installed.Please check the Installation Guide.')

    @abstractmethod
    def select(self, competences):
        """Select the most competent classifier for
        the classification of the query sample x.
        The most competent classifier (dcs) or an ensemble
        with the most competent classifiers (des) is returned

        Parameters
        ----------
        competences : array of shape = [n_samples, n_classifiers]
                      The estimated competence level of each base classifier
                      for test example

        Returns
        -------
        selected_classifiers : array containing the selected base classifiers
                               for each test sample

        """
        pass

    @abstractmethod
    def estimate_competence(self, query, neighbors, distances=None,
                            predictions=None):
        """estimate the competence of each base classifier :math:`c_{i}`
        the classification of the query sample :math:`\\mathbf{x}`.
        Returns an array containing the level of competence estimated
        for each base classifier. The size of the vector is equals to
        the size of the generated_pool of classifiers.

        Parameters
        ----------
        query : array of shape  = [n_samples, n_features]
                The query sample

        neighbors : array of shale = [n_samples, n_neighbors]
                    Indices of the k nearest neighbors according for each
                    test sample.

        distances : array of shale = [n_samples, n_neighbors]
                    Distances of the k nearest neighbors according for each
                    test sample.

        predictions : array of shape = [n_samples, n_classifiers]
                      Predictions of the base classifiers for all test examples
        Returns
        -------
        competences : array = [n_classifiers] containing the competence level
                      estimated for each base classifier
        """
        pass

    @abstractmethod
    def classify_with_ds(self, query, predictions, probabilities=None,
                         neighbors=None, distances=None, DFP_mask=None):
        """Predicts the label of the corresponding query sample.
        Returns the predicted label.

        Parameters
        ----------
        query : array of shape  = [n_samples, n_features]
            The test examples.

        predictions : array of shape = [n_samples, n_classifiers]
            Predictions of the base classifiers for all test examples

        probabilities : array of shape = [n_samples, n_classifiers, n_classes]
            Probabilities estimates of each base classifier for all test
            examples (For methods that always require probabilities from the
            base classifiers)

        neighbors : array of shale = [n_samples, n_neighbors]
            Indices of the k nearest neighbors according for each test sample

        distances : array of shale = [n_samples, n_neighbors]
            Distances of the k nearest neighbors according for each test sample

        DFP_mask : array of shape = [n_samples, n_classifiers]
            Mask containing 1 for the selected base classifier and 0 otherwise.

        Returns
        -------
        predicted_label : array of shape = [n_samples]
            The predicted label for each query
        """
        pass

    @abstractmethod
    def predict_proba_with_ds(self, query, predictions, probabilities,
                              neighbors=None, distances=None, DFP_mask=None):
        """Predicts the posterior probabilities of the corresponding
        query sample. Returns the probability estimates of each class.

        Parameters
        ----------
        query : array of shape  = [n_samples, n_features]
            The test examples.

        predictions : array of shape = [n_samples, n_classifiers]
            Predictions of the base classifiers for all test examples

        probabilities : array of shape = [n_samples, n_classifiers, n_classes]
            The predictions of each base classifier for all samples (For
            methods that always require probabilities from the base
            classifiers).

        neighbors : array of shape = [n_samples, n_neighbors]
            Indices of the k nearest neighbors according for each test sample

        distances : array of shale = [n_samples, n_neighbors]
            Distances of the k nearest neighbors according for each test sample

        DFP_mask : array of shape = [n_samples, n_classifiers]
           Mask containing 1 for the selected base classifier and 0 otherwise.

        Returns
        -------
        predicted_proba: array of shape = [n_samples, n_classes]
            Posterior probabilities estimates for each test example.
        """
        pass

    def fit(self, X, y):
        """Prepare the DS model by setting the KNN algorithm and
        pre-processing the information required to apply the DS
        methods

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The input data.

        y : array of shape = [n_samples]
            class labels of each example in X.

        Returns
        -------
        self
        """
        self.random_state_ = check_random_state(self.random_state)

        # Check if the length of X and y are consistent.
        X, y = check_X_y(X, y)

        # Check if the pool of classifiers is None.
        # If yes, use a BaggingClassifier for the pool.
        if self.pool_classifiers is None:
            self.pool_classifiers_ = BaggingClassifier(
                random_state=self.random_state_)
            self.pool_classifiers_.fit(X, y)
        else:
            self._check_base_classifier_fitted()
            self.pool_classifiers_ = self.pool_classifiers

        self.n_classifiers_ = len(self.pool_classifiers_)

        # check if the input parameters are correct. Raise an error if the
        # generated_pool is not fitted or k < 1
        self._validate_parameters()

        # Check if base classifiers are not using LabelEncoder (the case for
        # scikit-learn's ensembles):
        if isinstance(self.pool_classifiers_, BaseEnsemble):
            self.base_already_encoded_ = True
        else:
            self.base_already_encoded_ = False

        y_ind = self._setup_label_encoder(y)
        self._set_dsel(X, y_ind)

        # validate the value of k
        self._validate_k()
        self._set_region_of_competence_algorithm()
        self._fit_region_competence(X, y_ind)

        # validate the IH
        if(self.with_IH):
            self._validate_ih()
        return self

    def _compute_highest_possible_IH(self):
        highest_IH = (self.safe_k - math.ceil(
            self.safe_k / self.n_classes_)) / self.safe_k
        return highest_IH

    def _validate_ih(self):
        highest_IH = self._compute_highest_possible_IH()
        if(self.IH_rate > highest_IH):
            warnings.warn("IH_rate is bigger than the highest possible IH.",
                          category=RuntimeWarning)

    def _validate_k(self):

        # validate safe_k
        if self.k is None:
            self.k_ = self.n_samples_
        elif self.k > self.n_samples_:
            msg = "k is bigger than DSEL size. Using All DSEL examples " \
                  "for competence estimation."
            warnings.warn(msg, category=RuntimeWarning)
            self.k_ = self.n_samples_ - 1
        else:
            self.k_ = self.k

        # Validate safe_k
        if self.with_IH and self.safe_k is None:
            self.safe_k = self.k

    def _setup_label_encoder(self, y):
        self.enc_ = LabelEncoder()
        y_ind = self.enc_.fit_transform(y)
        self.classes_ = self.enc_.classes_

        return y_ind

    def _encode_base_labels(self, y):
        if self.base_already_encoded_:
            return y
        else:
            return self.enc_.transform(y)

    def _fit_region_competence(self, X, y):
        """Fit the k-NN classifier inside the dynamic selection method.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The Input data.

        y : array of shape = [n_samples]
            class labels of each sample in X.

        """
        self.roc_algorithm_.fit(X, y)

    def _set_dsel(self, X, y):
        """Pre-Process the input X and y data into the dynamic selection
        dataset(DSEL) and get information about the structure of the data
        (e.g., n_classes, N_samples, classes)

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The Input data.

        y : array of shape = [n_samples]
            class labels of each sample in X.
        """
        self.DSEL_data_ = X
        self.DSEL_target_ = y
        self.n_classes_ = self.classes_.size
        self.n_features_ = X.shape[1]
        self.n_samples_ = self.DSEL_target_.size
        self.DSEL_processed_, self.BKS_DSEL_ = self._preprocess_dsel()

    def _set_region_of_competence_algorithm(self):
        if self.knn_classifier is None:
            self.knn_class_ = functools.partial(KNeighborsClassifier,
                                                n_jobs=-1,
                                                algorithm="auto")
        elif isinstance(self.knn_classifier, str):
            if self.knn_classifier == "faiss":
                self.knn_class_ = functools.partial(
                    faiss_knn_wrapper.FaissKNNClassifier,
                    n_jobs=-1, algorithm="auto")
            elif self.knn_classifier == "knn":
                self.knn_class_ = functools.partial(KNeighborsClassifier,
                                                    n_jobs=-1,
                                                    algorithm="auto")
            else:
                raise ValueError(
                    '"knn_classifier" should be one of the following '
                    '["knn", "faiss"] or an estimator class')

        elif callable(self.knn_classifier):

            self.knn_class_ = self.knn_classifier

        else:
            raise ValueError('"knn_classifier" should be one of the following '
                             '["knn", "faiss"] or an estimator class')
        self.roc_algorithm_ = self.knn_class_(self.k)

    def _get_region_competence(self, query, k=None):
        """Compute the region of competence of the query sample
        using the data belonging to DSEL.

        Parameters
        ----------
        query : array of shape = [n_samples, n_features]
                The test examples.

        k : int (Default = self.k)
            The number of neighbors used to in the region of competence.

        Returns
        -------
        dists : list of shape = [n_samples, k]
                The distances between the query and each sample in the region
                of competence. The vector is ordered in an ascending fashion.

        idx : list of shape = [n_samples, k]
              Indices of the instances belonging to the region of competence of
              the given query sample.
        """
        if k is None:
            k = self.k_

        dists, idx = self.roc_algorithm_.kneighbors(query, n_neighbors=k,
                                                    return_distance=True)

        return np.atleast_2d(dists), np.atleast_2d(idx)

    def predict(self, X):
        """Predict the class label for each sample in X.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The input data.

        Returns
        -------
        predicted_labels : array of shape = [n_samples]
                           Predicted class label for each sample in X.
        """
        # Check if the DS model was trained
        check_is_fitted(self,
                        ["DSEL_processed_", "DSEL_data_", "DSEL_target_"])

        # Check if X is a valid input
        X = check_array(X)
        self._check_num_features(X)

        n_samples = X.shape[0]
        predicted_labels = np.empty(n_samples, dtype=np.intp)

        if self.needs_proba:
            base_probabilities = self._predict_proba_base(X)
            base_predictions = base_probabilities.argmax(axis=2)
        else:
            base_probabilities = None
            base_predictions = self._predict_base(X)

        all_agree_vector = BaseDS._all_classifier_agree(base_predictions)
        ind_all_agree = np.where(all_agree_vector)[0]

        # Since the predictions are always the same, get the predictions of the
        # first base classifier.
        if ind_all_agree.size:
            predicted_labels[ind_all_agree] = base_predictions[
                ind_all_agree, 0]

        # For the samples with disagreement, perform the dynamic selection
        # steps. First step is to collect the samples with disagreement
        # between base classifiers
        ind_disagreement = np.where(~all_agree_vector)[0]
        if ind_disagreement.size:

            X_DS = X[ind_disagreement, :]

            # If the method is based on clustering and does not use IH there
            # is no need to compute the Neighbors
            if hasattr(self, "clustering_") and not self.with_IH:
                distances = neighbors = None
            else:
                # Then, we estimate the nearest neighbors for all samples that
                # we need to call DS routines
                distances, neighbors = self._get_region_competence(X_DS)

            if self.with_IH:
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
                    ind_knn_original_matrix = ind_disagreement[
                        ind_knn_classifier]
                    prediction_knn, _ = mode(y_neighbors, axis=1)
                    predicted_labels[
                        ind_knn_original_matrix] = prediction_knn.reshape(-1, )

                    # Remove from the neighbors and distance matrices the
                    # samples that were classified using the KNN
                    neighbors = np.delete(neighbors, ind_knn_classifier,
                                          axis=0)
                    distances = np.delete(distances, ind_knn_classifier,
                                          axis=0)
            else:
                # IH was not considered. So all samples with disagreement are
                # passed down to the DS algorithm
                ind_ds_classifier = np.arange(ind_disagreement.size)

            # At this stage the samples which all base classifiers agrees or
            # that are associated with low hardness were already classified.
            # The remaining samples are now passed down to the DS techniques
            # for classification.

            #  First check whether there are still samples to be classified.
            if ind_ds_classifier.size:

                # IF the DFP pruning is considered, calculate the DFP mask
                # for all samples in X
                if self.DFP:
                    DFP_mask = self._frienemy_pruning(neighbors)
                else:
                    DFP_mask = np.ones(
                        (ind_ds_classifier.size, self.n_classifiers_))

                # Get the real indices_ of the samples that will be classified
                # using a DS algorithm.
                ind_ds_original_matrix = ind_disagreement[ind_ds_classifier]

                if self.needs_proba:
                    selected_probabilities = base_probabilities[
                        ind_ds_original_matrix]
                else:
                    selected_probabilities = None

                pred_ds = self.classify_with_ds(X_DS[ind_ds_classifier],
                                                base_predictions[
                                                    ind_ds_original_matrix],
                                                selected_probabilities,
                                                neighbors=neighbors,
                                                distances=distances,
                                                DFP_mask=DFP_mask)
                predicted_labels[ind_ds_original_matrix] = pred_ds

        return self.classes_.take(predicted_labels)

    def predict_proba(self, X):
        """Estimates the posterior probabilities for sample in X.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The input data.

        Returns
        -------
        predicted_proba : array of shape = [n_samples, n_classes]
                          Probabilities estimates for each sample in X.
        """
        # Check if the DS model was trained
        check_is_fitted(self,
                        ["DSEL_processed_", "DSEL_data_", "DSEL_target_"])

        # Check if X is a valid input
        X = check_array(X, ensure_2d=False)

        # Check if the base classifiers are able to estimate posterior
        # probabilities (implements predict_proba method).
        self._check_predict_proba()

        base_probabilities = self._predict_proba_base(X)
        base_predictions = base_probabilities.argmax(axis=2)

        n_samples = X.shape[0]
        predicted_proba = np.zeros((n_samples, self.n_classes_))

        all_agree_vector = BaseDS._all_classifier_agree(base_predictions)
        ind_all_agree = np.where(all_agree_vector)[0]

        if ind_all_agree.size:
            predicted_proba[ind_all_agree] = base_probabilities[
                ind_all_agree].mean(axis=1)

        ind_disagreement = np.where(~all_agree_vector)[0]

        if ind_disagreement.size:
            X_DS = X[ind_disagreement, :]

            # Always calculating the neighborhood. Passing that to classify
            # later
            # TODO: Check problems with DES Clustering method. Maybe add a
            # check to prevent that here. (or do clustering instead)
            # Then, we estimate the nearest neighbors for all samples that we
            # need to call DS routines
            distances, neighbors = self._get_region_competence(X_DS)

            if self.with_IH:
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

                    # Accessing which samples in the original matrix are
                    # associated with the low instance hardness indices_.
                    ind_knn_original_matrix = ind_disagreement[
                        ind_knn_classifier]

                    predicted_proba[ind_knn_original_matrix] = \
                        self.roc_algorithm_.predict_proba(
                            X_DS[ind_knn_classifier])

                    # Remove from the neighbors and distance matrices the
                    # samples that were classified using the KNN
                    neighbors = np.delete(neighbors, ind_knn_classifier,
                                          axis=0)
                    distances = np.delete(distances, ind_knn_classifier,
                                          axis=0)
            else:
                # IH was not considered. So all samples with disagreement are
                # passed down to the DS algorithm
                ind_ds_classifier = np.arange(ind_disagreement.size)

            if ind_ds_classifier.size:
                # Check if the dynamic frienemy pruning should be used
                if self.DFP:
                    DFP_mask = self._frienemy_pruning(neighbors)
                else:
                    DFP_mask = np.ones(
                        (ind_ds_classifier.size, self.n_classifiers_))

                ind_ds_original_matrix = ind_disagreement[ind_ds_classifier]

                proba_ds = self.predict_proba_with_ds(
                    X[ind_ds_original_matrix],
                    base_predictions[
                        ind_ds_original_matrix],
                    base_probabilities[
                        ind_ds_original_matrix],
                    neighbors=neighbors,
                    distances=distances,
                    DFP_mask=DFP_mask)

                predicted_proba[ind_ds_original_matrix] = proba_ds

        return predicted_proba

    def _frienemy_pruning(self, neighbors):
        """Implements the Online Pruning method (frienemy) to remove base
        classifiers that do not cross the region of competence. We consider
        that a classifier crosses the region of competence if it correctly
        classify at least one sample for each different class in the region.

        Returns
        -------
        DFP_mask : array of shape = [n_samples, n_classifiers]
                   Mask containing 1 for the selected base classifier and 0
                   otherwise.

        neighbors : array of shale = [n_samples, n_neighbors]
                    indices of the k nearest neighbors according to each
                    instance

        References
        ----------
        Oliveira, D.V.R., Cavalcanti, G.D.C. and Sabourin, R., Online Pruning
        of Base Classifiers for Dynamic Ensemble Selection,
        Pattern Recognition, vol. 72, December 2017, pp 44-58.
        """
        # using a for loop for processing a batch of samples temporarily.
        # Change later to numpy processing
        if neighbors.ndim < 2:
            neighbors = np.atleast_2d(neighbors)

        n_samples, _ = neighbors.shape
        mask = np.zeros((n_samples, self.n_classifiers_))

        for sample_idx in range(n_samples):
            # Check if query is in a indecision region
            neighbors_y = self.DSEL_target_[
                neighbors[sample_idx, :self.safe_k]]

            if len(set(neighbors_y)) > 1:
                # There are more than on class in the region of competence
                # (So it is an indecision region).

                # Check if the base classifier predict the correct label for
                # a sample belonging to each class.
                for clf_index in range(self.n_classifiers_):
                    predictions = self.DSEL_processed_[
                        neighbors[sample_idx, :self.safe_k], clf_index]
                    correct_class_pred = [self.DSEL_target_[index] for
                                          count, index in
                                          enumerate(neighbors[sample_idx,
                                                    :self.safe_k])
                                          if predictions[count] == 1]

                    # If that is true, it means that it correctly classified
                    # at least one neighbor for each class in
                    # the region of competence
                    if np.unique(correct_class_pred).size > 1:
                        mask[sample_idx, clf_index] = 1.0
                # Check if all classifiers were pruned
                if not np.count_nonzero(mask[sample_idx, :]):
                    # Do not apply the pruning mechanism.
                    mask[sample_idx, :] = 1.0

            else:
                # The sample is located in a safe region. All base classifiers
                # can predict the label
                mask[sample_idx, :] = 1.0

        return mask

    def _preprocess_dsel(self):
        """Compute the prediction of each base classifier for
        all samples in DSEL. Used to speed-up the test phase, by
        not requiring to re-classify training samples during test.

        Returns
        -------
        DSEL_processed_ : array of shape = [n_samples, n_classifiers].
                         Each element indicates whether the base classifier
                         predicted the correct label for the corresponding
                         sample (True), otherwise (False).

        BKS_DSEL_ : array of shape = [n_samples, n_classifiers]
                   Predicted labels of each base classifier for all samples
                   in DSEL.
        """
        BKS_dsel = self._predict_base(self.DSEL_data_)
        processed_dsel = BKS_dsel == self.DSEL_target_[:, np.newaxis]

        return processed_dsel, BKS_dsel

    def _predict_base(self, X):
        """ Get the predictions of each base classifier in the pool for all
            samples in X.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The test examples.

        Returns
        -------
        predictions : array of shape = [n_samples, n_classifiers]
                      The predictions of each base classifier for all samples
                      in X.
        """
        predictions = np.zeros((X.shape[0], self.n_classifiers_),
                               dtype=np.intp)

        for index, clf in enumerate(self.pool_classifiers_):
            labels = clf.predict(X)
            predictions[:, index] = self._encode_base_labels(labels)
        return predictions

    def _predict_proba_base(self, X):
        """ Get the predictions (probabilities) of each base classifier in the
        pool for all samples in X.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The test examples.

        Returns
        -------
        probabilities : array of shape = [n_samples, n_classifiers, n_classes]
                        Probabilities estimates of each base classifier for all
                        test samples.
        """
        probabilities = np.zeros(
            (X.shape[0], self.n_classifiers_, self.n_classes_))

        for index, clf in enumerate(self.pool_classifiers_):
            probabilities[:, index] = clf.predict_proba(X)
        return probabilities

    def _preprocess_dsel_scores(self):
        """Compute the output profiles of the dynamic selection dataset (DSEL)
         Each position of the output profiles vector is the score obtained by a
         base classifier :math:`c_{i}`
         for the classes of the input sample.

        Returns
        -------
        scores : array of shape = [n_samples, n_classifiers, n_classes]
                 Scores (probabilities) for each class obtained by each base
                 classifier in the generated_pool
                 for each sample in X.
        """
        scores = np.empty(
            (self.n_samples_, self.n_classifiers_, self.n_classes_))
        for index, clf in enumerate(self.pool_classifiers_):
            scores[:, index, :] = clf.predict_proba(self.DSEL_data_)

        return scores

    @staticmethod
    def _all_classifier_agree(predictions):
        """Check whether there is a difference in opinion among the classifiers
        in the generated_pool.

        Parameters
        ----------
        predictions : array of shape = [n_samples, n_classifiers]
                      Predictions of the base classifiers for the test examples

        Returns
        -------
        array of shape = [classes] containing True if all classifiers in the
        generated_pool
                          agrees on the same label, otherwise False.
        """
        return np.all(predictions == predictions[:, 0].reshape(-1, 1), axis=1)

    def _validate_parameters(self):
        """Verify if the input parameters are correct (generated_pool and k)
        raises an error if k < 1 or generated_pool is not fitted.
        """
        if self.k is not None:
            if not isinstance(self.k, int):
                raise TypeError("parameter k should be an integer")
            if self.k <= 1:
                raise ValueError("parameter k must be higher than 1."
                                 "input k is {} ".format(self.k))

        if self.safe_k is not None:
            if not isinstance(self.safe_k, int):
                raise TypeError("parameter safe_k should be an integer")
            if self.safe_k <= 1:
                raise ValueError("parameter safe_k must be higher than 1."
                                 "input safe_k is {} ".format(self.safe_k))

        # safe_k should be equals or lower the neighborhood size k.
        if self.safe_k is not None and self.k is not None:
            if self.safe_k > self.k:
                raise ValueError(
                    "parameter safe_k must be equal or less than parameter k."
                    "input safe_k is {} and k is {}".format(self.k,
                                                            self.safe_k))

        if not isinstance(self.IH_rate, float):
            raise TypeError(
                "parameter IH_rate should be a float between [0.0, 0.5]")

        if self.IH_rate < 0 or self.IH_rate > 0.5:
            raise ValueError("Parameter IH_rate should be between [0.0, 0.5]."
                             "IH_rate = {}".format(self.IH_rate))
        self._validate_pool()

    def _validate_pool(self):
        """ Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute.

        Raises
        -------
        ValueError
            If the pool of classifiers is empty.
        """
        if self.n_classifiers_ <= 0:
            raise ValueError("n_classifiers must be greater than zero, "
                             "got {}.".format(self.n_classifiers_))

    def _check_num_features(self, X):
        """ Verify if the number of features (n_features) of X is equals to
        the number of features used to fit the model. Raises an error if
        n_features is different.

        Parameters
        ----------
        X : array of shape = [classes, n_features]
            The input data.

        Raises
        -------
        ValueError
            If X has a different dimensionality than the training data.
        """
        n_features = X.shape[1]
        if self.n_features_ != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features_ is {} and "
                             "input n_features_ is {} ".format(
                                self.n_features_,
                                n_features))

    def _check_predict_proba(self):
        """ Checks if each base classifier in the pool implements the
        predict_proba method.

        Raises
        -------
        ValueError
            If the base classifiers do not implements the predict_proba method.
        """
        for clf in self.pool_classifiers_:
            if "predict_proba" not in dir(clf):
                raise ValueError(
                    "All base classifiers should output probability estimates")

    def _check_base_classifier_fitted(self):
        """ Checks if each base classifier in the pool is fitted.

        Raises
        -------
        NotFittedError: If any of the base classifiers is not yet fitted.
        """
        for clf in self.pool_classifiers:
            check_is_fitted(clf, "classes_")
