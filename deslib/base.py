# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause


from abc import abstractmethod, ABCMeta

import numpy as np
from scipy.stats import mode
from sklearn.base import ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.preprocessing import LabelEncoder
from deslib.util.instance_harndess import hardness_region_competence
from deslib.util.aggregation import predict_proba_ensemble

from sklearn.ensemble import BaseEnsemble


class DS(ClassifierMixin):
    """Base class for a dynamic classifier (dcs) and ensemble (des) selection methods.

    All dcs and des techniques should inherit from this class.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, pool_classifiers, k=7, DFP=False, with_IH=False, safe_k=None, IH_rate=0.30):

        self.pool_classifiers = pool_classifiers
        self.n_classifiers = len(self.pool_classifiers)
        self.k = k
        self.DFP = DFP              # Dynamic Frienemy Pruning
        self.with_IH = with_IH      # Whether to use hardness to switch between DS and KNN
        self.safe_k = safe_k          # K value used for defining a safe region
        self.IH_rate = IH_rate
        self.processed_dsel = None
        self.BKS_dsel = None
        self.dsel_scores = None
        self.roc_algorithm = None   # Algorithm used to define the region of competence
        self.DSEL_data = None
        self.DSEL_target = None
        self.classes = None
        self.n_classes = None
        self.n_samples = None
        self.n_features = None
        self.neighbors = None
        self.distances = None
        self.DFP_mask = None       # Mask used to apply the classifier pruning

        if self.with_IH and self.safe_k is None:
            self.safe_k = self.k

        # check if the input parameters are correct. Raise an error if the generated_pool is not fitted or k < 1
        self._check_parameters()

        # Check if base classifiers are not using LabelEncoder (the case for scikit-learn's ensembles):
        if isinstance(self.pool_classifiers, BaseEnsemble):
            self.base_already_encoded = True
        else:
            self.base_already_encoded = False

    @abstractmethod
    def select(self, query):
        """Select the most competent classifier for
        the classification of the query sample x.
        The most competent classifier (dcs) or an ensemble
        with the most competent classifiers (des) is returned

        Parameters
        ----------
        query : array containing the test sample = [n_features]

        Returns
        -------
        indices : list with the indices of the selected base classifier(s)

        competences: array with competence estimates for all base classifier
        """
        pass

    @abstractmethod
    def estimate_competence(self, query, predictions=None):
        """estimate the competence of each base classifier ci
        the classification of the query sample x.
        Returns an array containing the level of competence estimated
        for each base classifier. The size of the vector is equals to
        the size of the generated_pool of classifiers.

        Parameters
        ----------
        query : array containing the test sample = [n_features]

        predictions : array of shape = [n_samples, n_classifiers]
                      Contains the predictions of all base classifier for all samples in the query array

        Returns
        -------
        competences : array = [n_classifiers] containing the competence level estimated
        for each base classifier
        """
        pass

    @abstractmethod
    def classify_instance(self, query, predictions):
        """Predicts the label of the corresponding query sample.
        Returns the predicted label.

        Parameters
        ----------
        query : array containing the test sample = [n_samples, n_features]

        predictions : array of shape = [n_samples, n_classifiers]
                      Contains the predictions of all base classifier for all samples in the query array

        Returns
        -------
        The predicted label of the query
        """
        pass

    @abstractmethod
    def predict_proba_instance(self, query):
        """Predicts the posterior probabilities of the corresponding query sample.
        Returns the probability estimates of each class.

        Parameters
        ----------
        query : array containing the test sample = [n_features]

        Returns
        -------
        predicted_proba : array = [n_classes] with the probability estimates for all classes
        """
        pass

    def fit(self, X, y):
        """Prepare the DS model by setting the KNN algorithm and
        pre-processing the information required to apply the DS
        methods

         Parameters
        ----------
        X : matrix of shape = [n_samples, n_features] with the data.

        y : class labels of each sample in X.

        Returns
        -------
        self
        """

        y_ind = self.setup_label_encoder(y)

        check_X_y(X, y_ind)
        self._set_dsel(X, y_ind)
        self._fit_region_competence(X, y_ind, self.k)
        return self

    def setup_label_encoder(self, y):
        self.enc = LabelEncoder()
        y_ind = self.enc.fit_transform(y)
        self.classes = self.enc.classes_

        return y_ind

    def _encode_base_labels(self, y):
        if self.base_already_encoded:
            return y
        else:
            return self.enc.transform(y)

    def _fit_region_competence(self, X, y, k):
        """Fit the k-NN classifier inside the dynamic selection method.
         Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The Input data.

        y : array of shape = [n_samples]
            class labels of each sample in X.

        k : int (Default=self.k)
            Number of neighbors used in the k-NN method
        """
        self.roc_algorithm = KNeighborsClassifier(n_neighbors=k, n_jobs=-1, algorithm='auto')
        self.roc_algorithm.fit(X, y)

    def _set_dsel(self, X, y):
        """Pre-Process the input X and y data into the
         dynamic selection dataset(DSEL). Also
         get information about the structure of the
         data (n_classes, N_samples, classes)

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The Input data.

        y : array of shape = [n_samples]
            class labels of each sample in X.
        """
        self.DSEL_data = X
        self.DSEL_target = y
        self.n_classes = self.classes.size
        self.n_features = X.shape[1]
        self.n_samples = self.DSEL_target.size
        self.processed_dsel, self.BKS_dsel = self._preprocess_dsel()

    def _get_region_competence(self, query, k=None):
        """Compute the region of competence of the query sample
        using the data belonging to DSEL.

        Returns
        -------
        dists : list of shape = [k]
                The distances between the query and each sample in the region of competence. The vector is ordered
                in an ascending fashion.

        idx : list of shape = [k]
              Indices of the instances belonging to the region of competence of the given query sample.
        """
        # Check if the neighborhood was already estimated to avoid unnecessary calculations.
        if self.distances is None or self.neighbors is None:
            if k is None:
                k = self.k

            dists, idx = self.roc_algorithm.kneighbors(query, n_neighbors=k, return_distance=True)

        else:
            dists = self.distances
            idx = self.neighbors

        return dists, idx

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
        self._check_is_fitted()
        # Check if X is a valid input
        self._check_input_predict(X)

        n_samples = X.shape[0]
        predicted_labels = np.empty(n_samples, dtype=np.intp)
        base_predictions = self._predict_base(X)
        all_agree_vector = self._all_classifier_agree(base_predictions)
        ind_all_agree = np.where(all_agree_vector)[0]

        # Since the predictions are always the same, get the predictions of the first base classifier.
        if ind_all_agree.size:
            predicted_labels[ind_all_agree] = base_predictions[ind_all_agree, 0]

        # For the samples with disagreement, perform the dynamic selection steps. First step is to collect the samples
        # with disagreement between base classifiers
        ind_disagreement = np.where(~all_agree_vector)[0]
        if ind_disagreement.size:

            X_DS = X[ind_disagreement, :]

            # Then, we estimate the nearest neighbors for all samples that we need to call DS routines
            self.distances, self.neighbors = self._get_region_competence(X_DS)

            if self.with_IH:
                # if IH is used, calculate the hardness level associated with each sample
                hardness = hardness_region_competence(self.neighbors, self.DSEL_target, self.safe_k)

                # Get the index associated with the low and hard samples. Samples with low hardness are passed down to the
                # knn classifier while samples with high hardness are passed down to the DS methods. So, here we split the
                # samples that are passed to down to each stage by calculating their indices.
                # TODO check how to do this part without needing two np.where (like one being complements of other)
                ind_knn_classifier = np.where(hardness <= self.IH_rate)[0]
                ind_ds_classifier = np.where(hardness > self.IH_rate)[0]

                if ind_knn_classifier.size:
                    # all samples with low hardness should be classified by the knn method here: First get the class associated
                    # with each neighbor
                    y_neighbors = self.DSEL_target[self.neighbors[ind_knn_classifier, :self.safe_k]]

                    # Accessing which samples in the original matrix are associated with the low instance hardness indices.
                    # This is important since the low hardness indices ind_knn_classifier was estimated based on a subset of
                    # samples
                    ind_knn_original_matrix = ind_disagreement[ind_knn_classifier]
                    prediction_knn, _ = mode(y_neighbors, axis=1)
                    predicted_labels[ind_knn_original_matrix] = prediction_knn.reshape(-1,)

                    # Remove from the neighbors and distance matrices the samples that were classified using the KNN
                    self.neighbors = np.delete(self.neighbors, ind_knn_classifier, axis=0)
                    self.distances = np.delete(self.distances, ind_knn_classifier, axis=0)
            else:
                # IH was not considered. So all samples with disagreement are passed down to the DS algorithm
                ind_ds_classifier = np.arange(ind_disagreement.size)

            # At this stage the samples which all base classifiers agrees or that are associated with low hardness
            # Were already classified. The remaining samples are now passed down to the DS techniques for classification.
            # First check whether there are still samples to be classified.
            if ind_ds_classifier.size:

                # IF the DFP pruning is considered, calculate the DFP mask for all samples in X
                if self.DFP:
                     self.DFP_mask = self._frienemy_pruning()
                else:
                    self.DFP_mask = np.ones((ind_ds_classifier.size, self.n_classifiers))

                # Get the real indices of the samples that will be classified using a DS algorithm.
                ind_ds_original_matrix = ind_disagreement[ind_ds_classifier]
                pred_ds = self.classify_instance(X_DS[ind_ds_classifier, :],
                                                 base_predictions[ind_ds_original_matrix, :])
                predicted_labels[ind_ds_original_matrix] = pred_ds

        self.neighbors = None
        self.distances = None

        return self.classes.take(predicted_labels)

        # for index, instance in enumerate(X):
        #     # Do not use dynamic selection if all base classifiers agrees on the
        #     # same label.
        #     instance = instance.reshape(1, -1)
        #     if self._all_classifier_agree(base_predictions[index]):
        #         predicted_labels[index] = base_predictions[index, 0]
        #     else:
        #         # proceeds with DS, calculates the region of competence of the query sample
        #         if self.DFP or self.with_IH:
        #             self.distances, self.neighbors = self._get_region_competence(instance)
        #
        #         # If Instance hardness (IH) is used, check the IH of the region of competence to decide between
        #         # DS or the roc_algorithm classifier
        #         if self.with_IH and (self._hardness_region_competence(self.neighbors) <= self.IH_rate):
        #             # use the KNN for prediction if the sample is located in a safe region.
        #             # predicted_labels[index] = self.roc_algorithm.predict(instance)
        #             # Using the pre-calculated set of neighbors to perform the decision
        #             y_neighbors = self.DSEL_target[self.neighbors[:self.safe_k]]
        #             predicted_labels[index], _ = mode(y_neighbors)
        #
        #         # Otherwise, use DS for classification
        #         else:
        #             # Check if the dynamic frienemy pruning should be used
        #             if self.DFP:
        #                     self.DFP_mask = self._frienemy_pruning()
        #             else:
        #                     self.DFP_mask = np.ones(self.n_classifiers)

        #             predicted_labels[index] = self.classify_instance(instance, base_predictions[index, :])
        #
        #         self.neighbors = None
        #         self.distances = None
        #
        # return self.classes.take(predicted_labels)

    def predict_proba(self, X):
        """Estimates the posterior probabilities for sample in X.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            The input data.

        Returns
        -------
        predicted_proba : array of shape = [n_samples, n_classes] with the
        probabilities estimates for each class in the classifier model.
        """
        # Check if the DS model was trained
        self._check_is_fitted()

        # Check if X is a valid input
        self._check_input_predict(X)

        # Check if the base classifiers are able to estimate posterior probabilities (implements predict_proba).
        self._check_predict_proba()

        n_samples = X.shape[0]
        predicted_proba = np.zeros((n_samples, self.n_classes))
        for index, instance in enumerate(X):
            # Do not use dynamic selection if all base classifiers agrees on the
            # same label.
            instance = instance.reshape(1, -1)
            if self._all_classifier_agree_query(instance):

                #  since it may have better probabilities estimates
                predicted_proba[index, :] = predict_proba_ensemble(self.pool_classifiers, instance)[0]

            else:

                # Check whether use the DFP or hardness information is used to compute the competence region
                if self.DFP or self.with_IH:
                    self.distances, self.neighbors = self._get_region_competence(instance)

                # If Instance hardness (IH) is used, check the hardness level of the region of competence
                # to decide between DS or the roc_algorithm classifier
                if self.with_IH and \
                        (hardness_region_competence(self.neighbors, self.DSEL_target, self.safe_k) <= self.IH_rate):
                    # use the KNN for prediction if the sample is located in a safe region.
                    # TODO: Optimize that to not re-calculate the neighborhood
                    predicted_proba[index, :] = self.roc_algorithm.predict_proba(instance)

                # Otherwise, use DS for classification
                else:
                    # Check if the dynamic frienemy pruning should be used
                    if self.DFP:
                            self.DFP_mask = self._frienemy_pruning()
                    else:
                            self.DFP_mask = np.ones(self.n_classifiers)

                    predicted_proba[index, :] = self.predict_proba_instance(instance)

        # Reset the neighbors and the distances as they are specific to a given query.
        self.neighbors = None
        self.distances = None
        return predicted_proba


    def _frienemy_pruning(self):
        """Implements the Online Pruning method (frienemy) to remove base classifiers that do not cross the
        region of competence. We consider that a classifier crosses the region of competence if it correctly classify
        at least one sample for each different class in the region.

        Returns
        -------
        DFP_mask : array = [n_classifiers] with the probability estimates for all classes

        Reference:
        -------
        Oliveira, D.V.R., Cavalcanti, G.D.C. and Sabourin, R., Online Pruning of Base Classifiers for Dynamic
        Ensemble Selection, Pattern Recognition, vol. 72, December 2017, pp 44-58.
        """
        # using a for loop for processing a batch of samples temporarily. Change later to numpy processing
        if self.neighbors.ndim < 2:
            self.neighbors = np.atleast_2d(self.neighbors)

        n_samples, n_neighbors = self.neighbors.shape
        mask = np.zeros((n_samples, self.n_classifiers))

        for sample_idx in range(n_samples):
            # Check if query is in a indecision region
            neighbors_y = self.DSEL_target[self.neighbors[sample_idx, :self.safe_k]]

            if len(set(neighbors_y)) > 1:
                # There are more than on class in the region of competence (So it is an indecision region).
                #mask = np.zeros(self.n_classifiers)

                # Check if the base classifier predict the correct label for a sample belonging to each class.
                for clf_index in range(self.n_classifiers):
                    predictions = self.processed_dsel[self.neighbors[sample_idx, :self.safe_k], clf_index]
                    correct_class_pred = [self.DSEL_target[index] for count, index in
                                          enumerate(self.neighbors[sample_idx, :self.safe_k])
                                          if predictions[count] == 1]
                    """
                    # If that is true, it means that it correctly classified at least one neighbor for each class in 
                    the region of competence
                    """
                    if np.unique(correct_class_pred).size > 1:
                        mask[sample_idx, clf_index] = 1.0
                # Check if all classifiers were pruned
                if not np.count_nonzero(mask[sample_idx, :]):
                    # Do not apply the pruning mechanism.
                    mask[sample_idx, :] = 1.0
                    #mask = np.ones(self.n_classifiers)
            else:
                # The sample is located in a safe region. All base classifiers can predict the label
                mask[sample_idx, :] = 1.0
                # mask = np.ones(self.n_classifiers)
        return mask

    def _get_classifier_ensemble(self, indices):
        """This function receive the indices of the selected classifiers and returns an ensemble with the selected
        base classifiers

        Returns
        -------
        classifier_ensemble : A list with the selected base classifiers
        """
        classifier_ensemble = [self.pool_classifiers[index] for index in indices]
        return classifier_ensemble

    def _preprocess_dsel(self):
        """Compute the prediction of each base classifier for
        all samples in DSEL. Used to speed-up the test phase, by
        not requiring to re-classify training samples during test.

        Returns
        -------
        processed_dsel : A matrix of size [num_samples, num_classifiers].
        Each element indicates whether the base classifier predicted the
        correct label for the corresponding sample (True) or not (False).
        Used to speed-up the testing time without requiring to classify
        """

        BKS_dsel = self._predict_base(self.DSEL_data)
        processed_dsel = BKS_dsel == self.DSEL_target[:, np.newaxis]

        return processed_dsel, BKS_dsel

    def _predict_base(self, X):
        predictions = np.zeros((X.shape[0], self.n_classifiers), dtype=np.intp)

        for index, clf in enumerate(self.pool_classifiers):
            labels = clf.predict(X)
            predictions[:, index] = self._encode_base_labels(labels)
        return predictions


    def _output_profile_transform(self, query):
        """Transform the query in an output profile. Each position of the output profiles vector
        is the score obtained by a base classifier ci for the classes of the query.

        Returns
        -------
        output_profile_query : The output profiles of the query data
        """
        output_profile_query = np.zeros(self.n_classifiers * self.n_classes)

        for index, clf in enumerate(self.pool_classifiers):

            scores = clf.predict_proba(query)
            output_profile_query[index * self.n_classes:(index * self.n_classes) + self.n_classes] = scores

        return output_profile_query

    def _preprocess_dsel_scores(self):
        """Compute the output profiles of the dynamic selection dataset (DSEL)
         Each position of the output profiles vector is the score obtained by a base classifier ci
         for the classes of the query.

        Returns
        -------
        dsel_scores : A matrix of size [num_samples, num_classifiers * num_class]
        containing the scores (probabilities) for each class
        obtained by each base classifier in the generated_pool for each sample in DSEL.
        """

        dsel_scores = np.zeros(
            (self.DSEL_target.size, self.n_classifiers * self.n_classes))

        for index, clf in enumerate(self.pool_classifiers):
            scores = clf.predict_proba(self.DSEL_data)
            dsel_scores[:, index * self.n_classes:(index * self.n_classes) + self.n_classes] = scores
        return dsel_scores

    def _check_predict_proba(self):
        for clf in self.pool_classifiers:
            check_is_fitted(clf, "classes_")
            if "predict_proba" not in dir(clf):
                raise ValueError("All base classifiers should output probability estimates")

    def _get_scores_dsel(self, clf_idx, sample_idx=None):
        """Get the outputs (scores) obtained by the base classifier
        for the selected samples in dsel

        Parameters
        ----------
        clf_idx : index of the base classifier

        sample_idx : index of the sample belonging to dsel.
        if sample_idx is not specified (None), the scores
        obtained for the whole dsel is returned

        Returns
        -------
        scores : scores obtained for the corresponding sample
        """
        if self.dsel_scores is None:
            raise NotFittedError('dsel_scores was not fitted yet. Call "_pre_process_dsel_scores" '
                                 'to pre-process the classification scores before its use.')

        if sample_idx is None:
            scores = self.dsel_scores[:, clf_idx * self.n_classes:(clf_idx * self.n_classes) + self.n_classes]
        else:
            scores = self.dsel_scores[sample_idx, clf_idx * self.n_classes:(clf_idx * self.n_classes) + self.n_classes]
        return scores

    def _all_classifier_agree(self, predictions):
        """Check whether there is a difference in opinion
        among the classifiers in the generated_pool.

        Parameters
        ----------
        predictions : array of shape = [n_samples, n_classifiers]
                      Matrix with the predictions of each base classifier for each sample.
        Returns
        -------
        array of shape = [n_samples] containing True if all classifiers in the generated_pool
        agrees on the same label, otherwise False for all samples
        """
        return np.all(predictions == predictions[:, 0].reshape(-1,1), axis=1)

    def _all_classifier_agree_query(self, query):
        """Check whether there is a difference in opinion
        among the classifiers in the generated_pool.

        Parameters
        ----------
        query : Array containing the query sample
        to be classified

        Returns
        -------
        True : if all classifiers in the generated_pool
        agrees on the same label. False otherwise
        """
        target = None
        for clf in self.pool_classifiers:
            [temp] = clf.predict(query)
            if target is None:
                target = temp

            if target != temp:
                return False

            target = temp

        return True

    def _check_parameters(self):
        """Verify if the input parameters are correct (generated_pool and k)
        raises an error if k < 1 or generated_pool is not fitted.
        """
        if self.k is not None:
            if not isinstance(self.k, int):
                raise TypeError("parameter k should be an integer")
            if self.k <= 1:
                raise ValueError("parameter k must be higher than 1."
                                 "input k is {} " .format(self.k))

        if self.safe_k is not None:
            if not isinstance(self.safe_k, int):
                raise TypeError("parameter safe_k should be an integer")
            if self.safe_k <= 1:
                raise ValueError("parameter safe_k must be higher than 1."
                                 "input safe_k is {} " .format(self.safe_k))

        # safe_k should be equals or lower the neighborhood size k.
        if self.safe_k is not None and self.k is not None:
            if self.safe_k > self.k:
                raise ValueError("parameter safe_k must be equal or less than parameter k."
                                 "input safe_k is {} and k is {}" .format(self.k, self.safe_k))

        if not isinstance(self.IH_rate, float):
            raise TypeError("parameter IH_rate should be a float between [0.0, 0.5]")

        if 0 > self.IH_rate or self.IH_rate > 0.5:
            raise ValueError("Parameter IH_rate should be between [0.0, 0.5]."
                             "IH_rate = {}" .format(self.IH_rate))

        self._validate_pool()

    def _check_is_fitted(self):
        """Verify if the dynamic selection algorithm was fitted.
        Raises an error if it is not fitted.
        """
        if self.roc_algorithm is None or self.processed_dsel is None:
            raise NotFittedError("DS method not fitted, "
                                 "call `fit` before exploiting the model.")

    def _validate_pool(self):
        """Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute."""
        if self.n_classifiers <= 0:
            raise ValueError("n_classifiers must be greater than zero, "
                             "got {}.".format(self.n_classifiers))

    def _check_num_features(self, X):
        """Verify if the number of features (n_features) of X is equals to the number
        of features used to fit the model. Raises an error if n_features is different.
        """
        n_features = X.shape[1]
        if self.n_features != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is {} and "
                             "input n_features is {} " .format(self.n_features, n_features))

    def _check_input_predict(self, X):

        if X is None or np.isnan(X).any():
            raise ValueError('The input argument X is invalid! X = {}' .format(X))

        if X.ndim != 2:
            if X.ndim < 2:
                raise Warning('A 1D array was passed down as input. It is being reshaped to a 2D array.')
            else:
                raise ValueError('The input argument X has an invalid shape. Shape X = {}' .format(X.shape))

        self._check_num_features(X)

