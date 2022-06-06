# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause


import functools
import math
import warnings
from abc import abstractmethod, ABCMeta

import numpy as np
from scipy.stats import mode
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import BaseEnsemble, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, normalize
from sklearn.utils.validation import (check_X_y, check_is_fitted, check_array,
                                      check_random_state)

from deslib.util import KNNE
from deslib.util import faiss_knn_wrapper
from deslib.util.dfp import frienemy_pruning_preprocessed
from deslib.util.instance_hardness import hardness_region_competence


class BaseDS(BaseEstimator, ClassifierMixin):
    """Base class for a dynamic classifier selection (dcs) and
       dynamic ensemble selection (des) methods.

    All DCS and DES techniques should inherit from this class.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, pool_classifiers=None, k=7, DFP=False, with_IH=False,
                 safe_k=None, IH_rate=0.30, needs_proba=False,
                 random_state=None, knn_classifier='knn',
                 knn_metric='minkowski', DSEL_perc=0.5, knne=False, n_jobs=-1,
                 voting=None):

        self.pool_classifiers = pool_classifiers
        self.k = k
        self.DFP = DFP
        self.with_IH = with_IH
        self.safe_k = safe_k
        self.IH_rate = IH_rate
        self.needs_proba = needs_proba
        self.random_state = random_state
        self.knn_classifier = knn_classifier
        self.knn_metric = knn_metric
        self.DSEL_perc = DSEL_perc
        self.knne = knne
        self.n_jobs = n_jobs
        self.voting = voting

        # Check optional dependency
        if knn_classifier == 'faiss' and not faiss_knn_wrapper.is_available():
            raise ImportError(
                'Using knn_classifier="faiss" requires that the FAISS library '
                'be installed.Please check the Installation Guide.')

    def fit(self, X, y):
        """Prepare the DS model by setting the KNN algorithm and
        pre-processing the information required to apply the DS
        methods

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input data.

        y : array of shape (n_samples)
            class labels of each example in X.

        Returns
        -------
        self
        """
        self.random_state_ = check_random_state(self.random_state)
        X, y = self._validate_data(
            X,
            y,
            accept_sparse="csr",
            dtype=np.float64,
            order="C",
            accept_large_sparse=False,
        )

        # Check if the pool of classifiers is None.
        # If yes, use a BaggingClassifier for the pool.
        if self.pool_classifiers is None:
            X_dsel, y_dsel = self._fit_pool_classifiers(X, y)
        else:
            self._check_base_classifier_fitted()
            self.pool_classifiers_ = self.pool_classifiers
            X_dsel = X
            y_dsel = y

        self.n_classifiers_ = len(self.pool_classifiers_)
        # allow base models with feature subspaces.
        if hasattr(self.pool_classifiers_, "estimators_features_"):
            self.estimator_features_ = \
                np.array(self.pool_classifiers_.estimators_features_)
        else:
            indices = np.arange(X.shape[1])
            self.estimator_features_ = np.tile(indices,
                                               (self.n_classifiers_, 1))

        # check if the input parameters are correct.
        self._setup_label_encoder(y)
        y_dsel = self.enc_.transform(y_dsel)
        self._set_dsel(X_dsel, y_dsel)
        self._set_region_of_competence_algorithm(X_dsel)
        self._validate_parameters()

        self.roc_algorithm_.fit(X_dsel, y_dsel)
        self.BKS_DSEL_ = self._predict_base(self.DSEL_data_)
        self.DSEL_processed_ = self.BKS_DSEL_ == y_dsel[:, np.newaxis]

        return self

    def get_competence_region(self, query, k=None):
        """Compute the region of competence of the query sample
        using the data belonging to DSEL.

        Parameters
        ----------
        query : array of shape (n_samples, n_features)
                The test examples.

        k : int (Default = self.k)
            The number of neighbors used to in the region of competence.

        Returns
        -------
        dists : array of shape (n_samples, k)
                The distances between the query and each sample in the region
                of competence. The vector is ordered in an ascending fashion.

        idx : array of shape (n_samples, k)
              Indices of the instances belonging to the region of competence of
              the given query sample.
        """
        if k is None:
            k = self.k_

        dists, idx = self.roc_algorithm_.kneighbors(query,
                                                    n_neighbors=k,
                                                    return_distance=True)

        return np.atleast_2d(dists), np.atleast_2d(idx)

    @abstractmethod
    def estimate_competence(self, competence_region, distances=None,
                            predictions=None):
        """estimate the competence of each base classifier :math:`c_{i}`
        the classification of the query sample :math:`\\mathbf{x}`.
        Returns an array containing the level of competence estimated
        for each base classifier. The size of the vector is equals to
        the size of the generated_pool of classifiers.

        Parameters
        ----------
        competence_region : array of shape (n_samples, n_neighbors)
                    Indices of the k nearest neighbors according for each
                    test sample.

        distances : array of shape (n_samples, n_neighbors)
                    Distances of the k nearest neighbors according for each
                    test sample.

        predictions : array of shape (n_samples, n_classifiers)
                      Predictions of the base classifiers for all test examples
        Returns
        -------
        competences : array (n_classifiers) containing the competence level
                      estimated for each base classifier
        """
        pass

    @abstractmethod
    def select(self, competences):
        """Select the most competent classifier for
        the classification of the query sample x.
        The most competent classifier (dcs) or an ensemble
        with the most competent classifiers (des) is returned

        Parameters
        ----------
        competences : array of shape (n_samples, n_classifiers)
                      The estimated competence level of each base classifier
                      for test example

        Returns
        -------
        selected_classifiers : array containing the selected base classifiers
                               for each test sample

        """
        pass

    @abstractmethod
    def classify_with_ds(self, predictions, probabilities=None,
                         neighbors=None, distances=None, DFP_mask=None):
        """Predicts the label of the corresponding query sample.
        Returns the predicted label.

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
        pass

    @abstractmethod
    def predict_proba_with_ds(self, predictions, probabilities,
                              neighbors=None, distances=None, DFP_mask=None):
        """Predicts the posterior probabilities of the corresponding
        query sample. Returns the probability estimates of each class.

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
        pass

    def predict(self, X):
        """Predict the class label for each sample in X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        predicted_labels : array of shape (n_samples)
                           Predicted class label for each sample in X.
        """
        X = self._check_predict(X)
        preds = np.empty(X.shape[0], dtype=np.intp)
        need_proba = self.needs_proba or self.voting == 'soft'

        base_preds, base_probas = self._preprocess_predictions(X, need_proba)
        # predict all agree
        ind_disagreement, ind_all_agree = self._split_agreement(base_preds)
        if ind_all_agree.size:
            preds[ind_all_agree] = base_preds[ind_all_agree, 0]
        # predict with IH
        if ind_disagreement.size:
            distances, ind_ds_classifier, neighbors = self._IH_prediction(
                X, ind_disagreement, preds, is_proba=False
            )
            # Predict with DS - Check if there are still samples to be labeled.
            if ind_ds_classifier.size:
                DFP_mask = self._get_DFP_mask(neighbors)
                inds, sel_preds, sel_probas = self._prepare_indices_DS(
                    base_preds, base_probas, ind_disagreement,
                    ind_ds_classifier)
                preds_ds = self.classify_with_ds(sel_preds, sel_probas,
                                                 neighbors, distances,
                                                 DFP_mask)
                preds[inds] = preds_ds

        return self.classes_.take(preds)

    def _check_predict(self, X):
        check_is_fitted(self,
                        ["DSEL_processed_", "DSEL_data_", "DSEL_target_"])
        X = check_array(X)
        if self.n_features_ != X.shape[1]:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {0} and "
                             "input n_features is {1}."
                             "".format(self.n_features_, X.shape[1]))
        return X

    def predict_proba(self, X):
        """Estimates the posterior probabilities for sample in X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The input data.

        Returns
        -------
        predicted_proba : array of shape (n_samples, n_classes)
                          Probabilities estimates for each sample in X.
        """
        X = self._check_predict(X)

        self._check_predict_proba()
        probas = np.zeros((X.shape[0], self.n_classes_))
        base_preds, base_probas = self._preprocess_predictions(X, True)
        # predict all agree
        ind_disagreement, ind_all_agree = self._split_agreement(base_preds)
        if ind_all_agree.size:
            probas[ind_all_agree] = base_probas[ind_all_agree].mean(axis=1)
        # predict with IH
        if ind_disagreement.size:
            distances, ind_ds_classifier, neighbors = self._IH_prediction(
                    X, ind_disagreement, probas, is_proba=True)
            # Predict with DS - Check if there are still samples to be labeled.
            if ind_ds_classifier.size:
                DFP_mask = self._get_DFP_mask(neighbors)
                inds, sel_preds, sel_probas = self._prepare_indices_DS(
                    base_preds, base_probas, ind_disagreement,
                    ind_ds_classifier)
                probas_ds = self.predict_proba_with_ds(sel_preds,
                                                       sel_probas,
                                                       neighbors, distances,
                                                       DFP_mask)
                probas[inds] = probas_ds
        return probas

    def _preprocess_predictions(self, X, req_proba):
        if req_proba:
            base_probabilities = self._predict_proba_base(X)
            base_predictions = base_probabilities.argmax(axis=2)
        else:
            base_probabilities = None
            base_predictions = self._predict_base(X)
        return base_predictions, base_probabilities

    def _split_agreement(self, base_predictions):
        all_agree_vector = BaseDS._all_classifier_agree(base_predictions)
        ind_all_agree = np.where(all_agree_vector)[0]
        ind_disagreement = np.where(~all_agree_vector)[0]
        return ind_disagreement, ind_all_agree

    def _IH_prediction(self, X, ind_disagree, predicted_proba, is_proba=False):
        X_DS = X[ind_disagree, :]
        distances, region_competence = self.get_competence_region(X_DS)
        if self.with_IH:
            ind_hard, ind_easy = self._split_easy_samples(region_competence)
            distances, region_competence = self._predict_easy_samples(
                X_DS, distances, ind_disagree, ind_easy,
                region_competence, predicted_proba, is_proba)
        else:
            # IH was not considered. So all samples go to predict with DS
            ind_hard = np.arange(ind_disagree.size)
        return distances, ind_hard, region_competence

    def _split_easy_samples(self, neighbors):
        hardness = hardness_region_competence(neighbors,
                                              self.DSEL_target_,
                                              self.safe_k)
        # Get the index associated with the easy and hard samples.
        # easy samples are classified by the knn.
        easy_samples_mask = hardness < self.IH_rate
        ind_knn_classifier = np.where(easy_samples_mask)[0]
        ind_ds_classifier = np.where(~easy_samples_mask)[0]
        return ind_ds_classifier, ind_knn_classifier

    def _predict_easy_samples(self, X_DS, distances, ind_disagreement,
                              ind_easy, neighbors, predictions, is_proba):
        if ind_easy.size:
            # Accessing which samples in the original array.
            ind_knn_original_matrix = ind_disagreement[ind_easy]

            if is_proba:
                predictions[ind_knn_original_matrix] = \
                    self.roc_algorithm_.predict_proba(
                        X_DS[ind_easy])
            else:
                y_neighbors = self.DSEL_target_[neighbors[ind_easy,
                                                :self.safe_k]]
                predictions_knn, _ = mode(y_neighbors, axis=1)
                predictions[ind_knn_original_matrix] = predictions_knn.reshape(
                    -1, )
            neighbors = np.delete(neighbors, ind_easy, axis=0)
            distances = np.delete(distances, ind_easy, axis=0)
        return distances, neighbors

    def _prepare_indices_DS(self, base_predictions, base_probabilities,
                            ind_disagreement, ind_ds_classifier):
        # Get the real indices_ of the samples that will be classified
        # using a DS algorithm.
        ind_ds_original_matrix = ind_disagreement[ind_ds_classifier]
        if base_probabilities is not None:
            selected_probas = base_probabilities[
                ind_ds_original_matrix]
        else:
            selected_probas = None
        selected_preds = base_predictions[ind_ds_original_matrix]
        return ind_ds_original_matrix, selected_preds, selected_probas

    def _get_DFP_mask(self, neighbors):
        if self.DFP:
            DFP_mask = frienemy_pruning_preprocessed(neighbors,
                                                     self.DSEL_target_,
                                                     self.DSEL_processed_)
        else:
            DFP_mask = np.ones((neighbors.shape[0], self.n_classifiers_))
        return DFP_mask

    def _fit_pool_classifiers(self, X, y):
        if len(X) < 2:
            raise ValueError('More than one sample is needed '
                             'if the pool of classifiers is not informed.')
        # Split the dataset into training (for the base classifier) and
        # DSEL (for DS)
        X_train, X_dsel, y_train, y_dsel = train_test_split(
            X, y, test_size=self.DSEL_perc,
            random_state=self.random_state_)
        self.pool_classifiers_ = BaggingClassifier(
            random_state=self.random_state_, n_jobs=self.n_jobs)
        self.pool_classifiers_.fit(X_train, y_train)
        return X_dsel, y_dsel

    def _check_label_encoder(self):
        # Check if base classifiers are not using LabelEncoder (the case for
        # scikit-learn's ensembles):
        if isinstance(self.pool_classifiers_, BaseEnsemble):
            if np.array_equal(self.pool_classifiers_.classes_,
                              self.pool_classifiers_[0].classes_):
                self.base_already_encoded_ = False
            else:
                self.base_already_encoded_ = True
        else:
            self.base_already_encoded_ = False

    def _compute_highest_possible_IH(self):
        highest_IH = (self.safe_k - math.ceil(
            self.safe_k / self.n_classes_)) / self.safe_k
        return highest_IH

    def _validate_ih(self):
        highest_IH = self._compute_highest_possible_IH()
        if self.IH_rate > highest_IH:
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
        self._check_label_encoder()
        self.enc_ = LabelEncoder()
        self.enc_.fit(y)
        self.classes_ = self.enc_.classes_

    def _encode_base_labels(self, y):
        if self.base_already_encoded_:
            return y
        else:
            return self.enc_.transform(y)

    def _set_dsel(self, X, y):
        """Pre-Process the input X and y data into the dynamic selection
        dataset(DSEL) and get information about the structure of the data
        (e.g., n_classes, n_samples, classes)

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The Input data.

        y : array of shape (n_samples)
            class labels of each sample in X.
        """
        self.DSEL_data_ = X
        self.DSEL_target_ = y
        self.n_classes_ = self.classes_.size
        self.n_features_ = X.shape[1]
        self.n_samples_ = self.DSEL_target_.size

    def _set_region_of_competence_algorithm(self, X):

        algorithm = "auto"
        metric_params = None

        if self.knn_metric == 'minkowski':
            metric = 'minkowski'
        elif self.knn_metric == 'mahalanobis':
            metric = 'mahalanobis'
            metric_params = {'VI': np.cov(X)}
            algorithm = "auto"
        else:
            raise ValueError('"knn_metric" should be one of the following '
                             '["minkowski", "mahalanobis"]')

        if self.knn_classifier is None or self.knn_classifier in ['knn',
                                                                  'sklearn']:
            knn_class = functools.partial(KNeighborsClassifier,
                                          n_jobs=self.n_jobs,
                                          algorithm=algorithm,
                                          metric=metric,
                                          metric_params=metric_params)
        elif self.knn_classifier == 'faiss':
            knn_class = functools.partial(
                faiss_knn_wrapper.FaissKNNClassifier,
                n_jobs=self.n_jobs, algorithm="brute")
        elif callable(self.knn_classifier):
            knn_class = self.knn_classifier
        else:
            raise ValueError('"knn_classifier" should be one of the following '
                             '["knn", "faiss", None] or an estimator class.')

        if self.knne:
            self.knn_class_ = functools.partial(
                KNNE,
                knn_classifier=knn_class,
                n_jobs=self.n_jobs,
                algorithm="auto")
        else:
            self.knn_class_ = knn_class

        self.roc_algorithm_ = self.knn_class_(n_neighbors=self.k)

    def _preprocess_dsel(self):
        """Compute the prediction of each base classifier for
        all samples in DSEL. Used to speed-up the test phase, by
        not requiring to re-classify training samples during test.

        Returns
        -------
        DSEL_processed_ : array of shape (n_samples, n_classifiers).
                         Each element indicates whether the base classifier
                         predicted the correct label for the corresponding
                         sample (True), otherwise (False).

        BKS_DSEL_ : array of shape (n_samples, n_classifiers)
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
        X : array of shape (n_samples, n_features)
            The test examples.

        Returns
        -------
        predictions : array of shape (n_samples, n_classifiers)
                      The predictions of each base classifier for all samples
                      in X.
        """
        predictions = np.zeros((X.shape[0], self.n_classifiers_),
                               dtype=np.intp)

        for index, clf in enumerate(self.pool_classifiers_):
            labels = clf.predict(X[:, self.estimator_features_[index]])
            predictions[:, index] = self._encode_base_labels(labels)
        return predictions

    def _predict_proba_base(self, X):
        """ Get the predictions (probabilities) of each base classifier in the
        pool for all samples in X.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
            The test examples.

        Returns
        -------
        probabilities : array of shape (n_samples, n_classifiers, n_classes)
                        Probabilities estimates of each base classifier for all
                        test samples.
        """
        probas = np.zeros(
            (X.shape[0], self.n_classifiers_, self.n_classes_))

        for index, clf in enumerate(self.pool_classifiers_):
            probas[:, index] = clf.predict_proba(
                X[:, self.estimator_features_[index]])
        return probas

    @staticmethod
    def _all_classifier_agree(predictions):
        """Check whether there is a difference in opinion among the classifiers
        in the generated_pool.

        Parameters
        ----------
        predictions : array of shape (n_samples, n_classifiers)
                      Predictions of the base classifiers for the test examples

        Returns
        -------
        array of shape (classes)
            containing True if all classifiers in the generated_pool agrees
            on the same label, otherwise False.
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

        self._validate_pool_classifiers()
        # validate the value of k
        self._validate_k()
        # validate the IH
        if self.with_IH:
            self._validate_ih()

    def _validate_pool_classifiers(self):
        """ Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute.

        Raises
        -------
        ValueError
            If the pool of classifiers is empty.
        """
        if self.n_classifiers_ <= 1:
            raise ValueError("n_classifiers must be greater than one, "
                             "got {}.".format(self.n_classifiers_))

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
