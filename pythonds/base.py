# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause


import collections
import numbers
from abc import abstractmethod, ABCMeta

import numpy as np
from sklearn.base import ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.neighbors import KNeighborsClassifier

from pythonds.util.prob_functions import softmax


class DS(ClassifierMixin):
    """Base class for a dynamic classifier (dcs) and ensemble (des) selection methods.

    All dcs and des techniques should inherit from this class.

    Warning: This class should not be used directly.
    Use derived classes instead.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self, pool_classifiers, k=7, DFP=False, with_IH=False, safe_k=None, IH_rate=0.30, aknn=False):

        self.pool_classifiers = pool_classifiers
        self.n_classifiers = len(self.pool_classifiers)
        self.k = k
        self.DFP = DFP              # Dynamic Frienemy Pruning
        self.aknn = aknn            # Adaptive K-NN
        self.with_IH = with_IH      # Use hardness to switch between DS and KNN
        self.IH_k = safe_k            # K for defining a safe region
        self.IH_rate = IH_rate
        self.processed_dsel = None
        self.BKS_dsel = None
        self.dsel_scores = None
        self.knn = None
        self.DSEL_data = None
        self.DSEL_target = None
        self.classes = None
        self.n_classes = None
        self.n_samples = None
        self.n_features = None
        self.neighbors = None
        self.distances = None
        self.mask = None

        if self.with_IH and self.Ih_k is None:
            self.IH_k = self.k

        # check if the input parameters are correct. Raise an error if the generated_pool is not fitted or k < 1
        self._check_parameters()

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
    def estimate_competence(self, query):
        """estimate the competence of each base classifier ci
        the classification of the query sample x.
        Returns an array containing the level of competence estimated
        for each base classifier. The size of the vector is equals to
        the size of the generated_pool of classifiers.

        Parameters
        ----------
        query : array containing the test sample = [n_features]

        Returns
        -------
        competences : array = [n_classifiers] containing the competence level estimated
        for each base classifier
        """
        pass

    @abstractmethod
    def classify_instance(self, query):
        """Predicts the label of the corresponding query sample.
        Returns the predicted label.

        Parameters
        ----------
        query : array containing the test sample = [n_features]

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
        self._set_dsel(X, y)
        self.fit_knn(X, y, self.k)
        return self

    # TODO: Implement the KNN-Equality (KNNE)
    def fit_knn(self, X, y, k):
        """Fit the k-NN classifier inside the dynamic selection method.
         Parameters
        ----------
        X : matrix of shape = [n_samples, n_features] with the data.

        y : class labels of each sample in X.

        k : int (Default=self.k), Number of neighbors calculated by the k-NN method
        """
        if self.aknn:
            # TODO see how can implement the adaptive KNN method.
            # changing the DSEL set prior to use standard KNN?)
            self.knn = KNeighborsClassifier(
                n_neighbors=k, n_jobs=-1, algorithm='auto')

        else:
            self.knn = KNeighborsClassifier(
                n_neighbors=k, n_jobs=-1, algorithm='auto')
        self.knn.fit(X, y)

    def _set_dsel(self, X, y):
        """Pre-Process the input X and y data into the
         dynamic selection dataset(DSEL). Also
         get information about the structure of the
         data (n_classes, N_samples, classes)

        Parameters
        ----------
        X : ndarray of shape = [n_samples, n_features] each row corresponding to a different instance

        y: array of shape = [n_samples] containing the labels of each instance
        """
        self.DSEL_data = X
        self.DSEL_target = y
        self.classes = np.unique(self.DSEL_target)
        self.n_classes = self.classes.size
        self.n_samples = self.DSEL_target.size
        self.processed_dsel, self.BKS_dsel = self._preprocess_dsel()

    def _get_region_competence(self, query, k=None):
        """Compute the region of competence of the query sample
        using the data belonging to DSEL.

        Returns
        -------
        dists : A vector with the distances between the query and
        each sample in the region of competence. The vector is ordered
        in an ascending fashion
        idx : Indices of the instances belonging to the region of competence
        of the query sample.
        """

        # Check if the neighborhood was already estimated to avoid unecessary calculations.
        if self.distances is None or self.neighbors is None:
            if k is None:
                k = self.k

            [dists], [idx] = self.knn.kneighbors(query, n_neighbors=k, return_distance=True)

        else:
            dists = self.distances
            idx = self.neighbors

        return dists, idx

    def predict(self, X):
        """Predict the label of each sample in X.
        returns the predicted label.

        Parameters
        ----------
        X : ndarray of shape = [n_samples, n_features]

        Returns
        -------
        predicted_labels : array of shape = [n_samples] with the
            predicted class for each sample.
        """
        n_samples = X.shape[0]
        predicted_labels = np.zeros(n_samples)
        for index, instance in enumerate(X):
            # Do not use dynamic selection if all base classifiers agrees on the
            # same label.
            if self._all_classifier_agree(instance):

                predicted_labels[index] = self.pool_classifiers[0].predict(instance.reshape(1, -1))[0]

            else:
                self.neighbors = None
                self.distances = None

                # proceeds with DS, calculates the region of competence of the query sample
                if self.DFP or self.with_IH:
                    self.distances, self.neighbors = self._get_region_competence(instance.reshape(1, -1))

                # If Instance hardness (IH) is used, check the IH of the region of competence to decide between
                # DS or the knn classifier
                if self.with_IH and (self._hardness_region_competence(self.neighbors) <= self.IH_rate):
                    # use the KNN for prediction if the sample is located in a safe region.
                    predicted_labels[index] = self.knn.predict(instance.reshape(1, -1))

                # Otherwise, use DS for classification
                else:
                    # Check if the dynamic frienemy pruning should be used
                    if self.DFP:
                            self.mask = self._frienemy_pruning()
                    else:
                            self.mask = np.ones(self.n_classifiers)

                    predicted_labels[index] = self.classify_instance(instance.reshape(1, -1))

        return predicted_labels

    def predict_proba(self, X):
        """Estimates the posterior probabilities for each class in the classification problem.

        The returned probability estimates for all classes are ordered by the
        label of classes.

        Parameters
        ----------
        X : ndarray of shape = [n_samples, n_features]

        Returns
        -------
        predicted_proba : ndarray of shape = [n_samples, n_classes] with the
        probabilities estimates for each class in the classifier model.
        """
        n_samples = X.shape[0]
        predicted_proba = np.zeros(n_samples, self.n_classes)
        for index, instance in enumerate(X):
            # Do not use dynamic selection if all base classifiers agrees on the
            # same label.
            if self._all_classifier_agree(instance):

                # TODO: This raises question whether the most competent classifier should be used,
                #  since it may have better probabilities estimates
                predicted_proba[index] = self.pool_classifiers[0].predict(instance.reshape(1, -1))[0]

            else:
                # proceeds with DS, calculates the region of competence of the query sample

                # Reset the neighbors and the distances as the query changed.
                self.neighbors = None
                self.distances = None

                # Check whether use the DFP or hardness information is used to compute the competence region
                if self.DFP or self.with_IH:
                    self.distances, self.neighbors = self._get_region_competence(instance.reshape(1, -1))

                # If Instance hardness (IH) is used, check the hardness level of the region of competence
                # to decide between DS or the knn classifier
                if self.with_IH and (self._hardness_region_competence(self.neighbors) <= self.IH_rate):
                    # use the KNN for prediction if the sample is located in a safe region.
                    # TODO: Optimize that to not re-calculate the neighborhood
                    predicted_proba[index, :] = self.knn.predict_proba(instance.reshape(1, -1))

                # Otherwise, use DS for classification
                else:
                    # Check if the dynamic frienemy pruning should be used
                    if self.DFP:
                            self.mask = self._frienemy_pruning()
                    else:
                            self.mask = np.ones(self.n_classifiers)

                    predicted_proba[index, :] = self.predict_proba_instance(instance.reshape(1, -1))

        return predicted_proba

    # TODO: remove the aggregation functions from this class. They should accept other ensemble methods as well
    def majority_voting(self, indices, query):
        """Performs a majority voting combination scheme between the base classifiers
        specified in the vector indices. Returns the label of the query sample as the
        most voted class.

        Parameters
        ----------
        indices : index of the base classifier to be used in the combination scheme

        query : Sample to be classified

        Returns
        -------
        predicted_label : The label of the query sample, predicted by the majority voting rule
        """
        # if indices is empty (no classifier is selected, use all of them)
        if len(indices) == 0:
            indices = list(range(self.n_classifiers))

        votes = [self.pool_classifiers[idx].predict(query)[0] for idx in indices]
        counter = collections.Counter(votes)
        predicted_label = counter.most_common()[0][0]

        return predicted_label

    # TODO: remove the aggregation functions from this class. They should accept other ensemble methods as well
    def weighted_majority_voting(self, indices, weights, query):
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
        if len(indices) == 0:
            indices = list(range(self.n_classifiers))
            weights = np.ones(self.n_classifiers)/self.n_classifiers

        w_votes = np.zeros(self.n_classes)
        for idx in indices:
            w_votes[self.pool_classifiers[idx].predict(query)[0]] += weights[idx]
        predicted_label = np.argmax(w_votes)

        return predicted_label

    # TODO: Probably I can write an universal routine which predicts the probabilities estimates for single and ensemble
    def predict_proba_ensemble(self, query, indices, weights=None):
        """Compute the posterior probabilities of each class for the query

        Parameters
        ----------
        indices : index of the base classifier to be used in the combination scheme

        weights : Weight of each classifier in the decision (Default=None)

        query : Sample to predict the probability estimates

        Returns
        -------
        predicted_proba : Probability estimates for each class
        """
        # if indices is empty (no classifier is selected, use all of them)
        if len(indices) == 0:
            indices = list(range(self.n_classifiers))
            weights = np.ones(self.n_classifiers)/self.n_classifiers

        proba = np.zeros(self.n_classes)

        for idx in indices:
            proba += self.pool_classifiers[idx].predict_proba(query) * weights[idx]
        predicted_proba = softmax(proba)

        return predicted_proba

    def _hardness_region_competence(self, region_competence):
        """Calculate the Instance hardness of the sample based on its neighborhood. The sample is deemed hard to
        classify when there is overlap between different classes in the region of competence.

        This hardness measure is used to select
        whether use DS or use the KNN for the classification of a given query sample

        Parameters
        ----------
        region_competence : array = [k] containing the indices of the samples belonging to the region of competence

        Returns
        -------
        hardness : float, with the calculated IH of the corresponding sample

        Reference
        ----------
        [1] Smith, M.R., Martinez, T. and Giraud-Carrier, C., 2014. An instance level analysis of data complexity.
        Machine learning, 95(2), pp.225-256
        """
        neighbors_y = [self.DSEL_target[index] for index in region_competence[:self.IH_k]]

        counter = collections.Counter(neighbors_y)
        # Get the number of examples of the majority class in the region of competence
        count_most_common = counter.most_common()[0][1]
        hardness = (self.k - count_most_common) / self.k

        return hardness

    def _frienemy_pruning(self):
        """Implemente the Online Pruning method (frienemy) to remove base classifiers that do not cross the
        region of competence. We consider that a classifier crosses the region of competence if it correctly classify
        at least one sample for each different class in the region.

        Returns
        -------
        mask : array = [n_classifiers] with the probability estimates for all classes

        Reference:
        -------
        [1] Oliveira, D.V.R., Cavalcanti, G.D.C. and Sabourin, R., Online Pruning of Base Classifiers for Dynamic
        Ensemble Selection, Pattern Recognition, vol. 72, December 2017, pp 44-58.
        """
        # Check if query is in a indecision region
        neighbors_y = [self.DSEL_target[index] for index in self.neighbors[:self.IH_k]]
        if len(set(neighbors_y)) > 1:
            # There are more than on class in the region of competence (It is an indecision region).
            mask = np.zeros(self.n_classifiers)

            # Check if the base classifier predict the correct label for a sample belonging to each class.
            for clf_index in range(self.n_classifiers):
                predictions = [self.processed_dsel[index][clf_index] for index in self.neighbors]
                correct_class_pred = [self.DSEL_target[index] for count, index in enumerate(self.neighbors)
                                      if predictions[count] == 1]
                """
                # If that is true, it means that it correctly classified at least one neighbor for each class in 
                the region of competence
                """
                if np.array_equal(np.unique(neighbors_y), np.unique(correct_class_pred)):
                    mask[clf_index] = 1
                else:
                    mask[clf_index] = 0
                if ~np.all(mask):
                    # Considers the whole pool if no base classifiers croesses the region of competence
                    mask = np.ones(self.n_classifiers)
                return mask
        else:
            # The sample is located in a safe region. All base classifiers can predict the label
            mask = np.ones(self.n_classifiers)
        return mask

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
        processed_dsel = np.zeros((self.DSEL_target.size, self.n_classifiers))
        BKS_dsel = np.zeros((self.DSEL_target.size, self.n_classifiers))

        for index, clf in enumerate(self.pool_classifiers):
            labels = clf.predict(self.DSEL_data)
            hit = (labels == self.DSEL_target)
            processed_dsel[:, index] = hit
            BKS_dsel[:, index] = labels
        return processed_dsel, BKS_dsel

    def _output_profile_transform(self, query):
        """Transform the query in an output profile. Each position of the ouput profiles vector
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
         Each position of the ouput profiles vector is the score obtained by a base classifier ci
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
        if sample_idx is None:
            scores = self.dsel_scores[:, clf_idx * self.n_classes:(clf_idx * self.n_classes) + self.n_classes]
        else:
            scores = self.dsel_scores[sample_idx, clf_idx * self.n_classes:(clf_idx * self.n_classes) + self.n_classes]
        return scores

    def _all_classifier_agree(self, query):
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
        query = query.reshape(1, -1)
        for clf in self.pool_classifiers:
            [temp] = clf.predict(query)
            if target is None:
                target = temp

            if target != temp:
                return False

            target = temp

        return True

    @staticmethod
    def _check_input_shape(X, y):
        """Check if the vectors X and y have the same number of samples. Raises an error
        if it is different.
        """
        if X.shape[0] != y.shape[0]:
            raise ValueError("The number of samples in X and y must be the same"
                             "n_samples X = %s, n_samples y = %s " % (X.shape[0], y.shape[0]))

    def _check_parameters(self):
        """Verify if the input parameters are correct (generated_pool and k)
        raises an error if k < 1 or generated_pool is not fitted.
        """
        if self.k is not None:
            if not isinstance(self.k, numbers.Integral):
                raise TypeError("parameter k should be an integer")
            if self.k <= 1:
                raise ValueError("parameter k must be higher than 1."
                                 "input k is %s " % self.k)

        # IH_h should be equals or lower the neighborhood size k
        if self.IH_k is not None and self.k is not None:
            if self.IH_k > self.k:
                raise ValueError("parameter IH_k must be equal or less than parameter k."
                                 "input IH_k is %s  and k is " % self.k, self.IH_k)
        self._validate_pool()

    def _check_is_fitted(self):
        """Verify if the dynamic selection algorithm was fitted.
        Raises an error if it is not fitted.
        """
        if self.knn is None:
            raise NotFittedError("Estimator not fitted, "
                                 "call `fit` before exploiting the model.")

    def _check_num_features(self, X):
        """Verify if the number of features (n_features) of X is equals to the number
        of features used to fit the model. Raises an error if n_features is different.
        """
        n_features = X.shape[1]
        if self.n_features != n_features:
            raise ValueError("Number of features of the model must "
                             "match the input. Model n_features is %s and "
                             "input n_features is %s "
                             % (self.n_features, n_features))

    def _validate_pool(self):
        """Check the estimator and the n_estimator attribute, set the
        `base_estimator_` attribute."""
        if not isinstance(self.n_classifiers, (numbers.Integral, np.int)):
            raise ValueError("n_classifiers must be an integer, "
                             "got {0}.".format(type(self.n_classifiers)))

        if self.n_classifiers <= 0:
            raise ValueError("n_classifiers must be greater than zero, "
                             "got {0}.".format(self.n_classifiers))

        for clf in self.pool_classifiers:
            if clf is None:
                raise ValueError("All base classifiers should be fitted before use")
            if "predict_proba" not in dir(clf):
                raise ValueError("All base classifiers should output probability estimates")
