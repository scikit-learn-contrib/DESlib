# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import warnings

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils.validation import check_is_fitted

from deslib.des.base import DES


class METADES(DES):
    """Meta learning for dynamic ensemble selection (META-DES).

    The META-DES framework is based on the assumption that the dynamic ensemble selection problem can be considered
    as a meta-problem. This meta-problem uses different criteria regarding the behavior of a base
    classifier :math:`c_{i}`, in order to decide whether it is competent enough to classify a given test
    sample.

    The framework performs a meta-training stage, in which, the meta-features are extracted from each instance
    belonging to the training and the dynamic selection dataset (DSEL). Then, the extracted meta-features are used
    to train the meta-classifier :math:`\\lambda`. The meta-classifier is trained to predict whether or not a
    base classifier :math:`c_{i}` is competent enough to classify a given input sample.

    When an unknown sample is presented to the system, the meta-features for each base classifier :math:`c_{i}`
    in relation to the input sample are calculated and presented to the meta-classifier. The meta-classifier
    estimates the competence level of the base classifier :math:`c_{i}` for the classification of the query sample.
    Base classifiers with competence level higher than a pre-defined threshold are selected. If no base classifier
    is selected, the whole pool is used for classification.

    Parameters
    ----------
    pool_classifiers : list of classifiers
                       The generated_pool of classifiers trained for the corresponding classification problem.
                       The classifiers should support methods "predict" and "predict_proba".

    k : int (Default = 7)
        Number of neighbors used to estimate the competence of the base classifiers.

    kp : int (Default = 5)
         Number of output profiles used to estimate the competence of the base classifiers.

    Hc : float (Default = 1.0)
         Sample selection threshold.

    selection_threshold : float(Default = 0.5)
                          Threshold used to select the base classifier. Only the base classifiers with competence
                          level higher than the selection_threshold are selected to compose the ensemble.

    mode : String (Default = "selection")
              Determines the mode of META-des that is used (selection, weighting or hybrid).

    DFP : Boolean (Default = False)
          Determines if the dynamic frienemy pruning is applied.

    with_IH : Boolean (Default = False)
              Whether the hardness level of the region of competence is used to decide between using the DS
              algorithm or the KNN for classification of a given query sample.

    safe_k : int (default = None)
             The size of the indecision region.

    IH_rate : float (default = 0.3)
              Hardness threshold. If the hardness level of the competence region is lower than
              the IH_rate the KNN classifier is used. Otherwise, the DS algorithm is used for classification.

    References
    ----------
    Cruz, R.M., Sabourin, R., Cavalcanti, G.D. and Ren, T.I., 2015. META-DES: A dynamic ensemble selection framework
    using meta-learning. Pattern Recognition, 48(5), pp.1925-1935.

    Cruz, R.M., Sabourin, R. and Cavalcanti, G.D., 2015, July. META-des. H: a dynamic ensemble selection technique
    using meta-learning and a dynamic weighting approach. In Neural Networks (IJCNN), 2015 International Joint
    Conference on (pp. 1-8).

    R. M. O. Cruz, R. Sabourin, and G. D. Cavalcanti, “Dynamic classifier selection: Recent advances and perspectives,”
    Information Fusion, vol. 41, pp. 195 – 216, 2018.

    """
    def __init__(self, pool_classifiers, meta_classifier=MultinomialNB(),
                 k=7,
                 kp=5,
                 Hc=1.0,
                 selection_threshold=0.5,
                 mode='selection',
                 DFP=False,
                 with_IH=False,
                 safe_k=None,
                 IH_rate=0.30):

        super(METADES, self).__init__(pool_classifiers, k, DFP=DFP,
                                      with_IH=with_IH, safe_k=safe_k, IH_rate=IH_rate,
                                      mode=mode, needs_proba=True)

        self._check_predict_proba()
        self._check_input_parameters(Hc, selection_threshold, meta_classifier)

        self.name = 'META-DES'
        self.Kp = kp
        self.Hc = Hc
        self.selection_threshold = selection_threshold

        if meta_classifier is None:
            warnings.warn("No classifier model passed for the Meta-Classifier. Using a Naive Bayes instead")
            self.meta_classifier = MultinomialNB()
        else:
            self.meta_classifier = meta_classifier

        self.op_knn = None
        self.n_meta_features = (self.k * 2) + self.Kp + 2

    def fit(self, X, y):
        """Prepare the DS model by setting the KNN algorithm and
        pre-processing the information required to apply the DS
        method.

        This method also extracts the meta-features and trains the meta-classifier :math:`\\lambda`
        if the meta-classifier was not yet trained.

        Parameters
        ----------
        X : array of shape = [n_samples, n_features]
            Data used to fit the model.

        y : array of shape = [n_samples]
            class labels of each example in X.

        Returns
        -------
        self
        """

        y_ind = self.setup_label_encoder(y)
        self._set_dsel(X, y_ind)
        self._fit_region_competence(X, y_ind, self.k)
        self.dsel_scores = self._preprocess_dsel_scores()

        # Reshape dsel_scores as a 2-D array for nearest neighbor calculations
        dsel_output_profiles = self.dsel_scores.reshape(self.n_samples, self.n_classifiers * self.n_classes)
        self._fit_OP(dsel_output_profiles, y_ind, self.Kp)

        # check whether the meta-classifier was already trained since it could have been pre-processed before
        try:
            check_is_fitted(self.meta_classifier, "classes_")
        except NotFittedError as _:
            # IF it is not fitted, generate the meta-training dataset and train the meta-classifier
            X_meta, y_meta = self._generate_meta_training_set()
            self._fit_meta_classifier(X_meta, y_meta)

        return self

    def _fit_OP(self, X_op, y_op, kp):
        """ Fit the set of output profiles.

        Parameters
        ----------
        X_op : array of shape = [n_samples, n_features]
               The output profiles of the Input data. n_features is equals to (n_classifiers x n_classes)

        y_op : array of shape = [n_samples]
               class labels of each sample in X_op.

        kp : int
             Number of output profiles used in the estimation.

        """
        self.op_knn = KNeighborsClassifier(n_neighbors=kp, n_jobs=-1, algorithm='auto')

        if self.n_classes == 2:
            # Get only the scores for one class since they are complementary
            X_temp = X_op[:, ::2]
            self.op_knn.fit(X_temp, y_op)
        else:
            self.op_knn.fit(X_op, y_op)

    def _sample_selection_agreement(self):
        """Check the number of base classifier that predict the correct label for the query sample.

        Returns
        -------
        pct_agree : array of shape [n_samples]
                    The percentage of the base classifier that predicted the correct label for each sample in DSEL.

        """
        # pct_agree = np.sum(self.processed_dsel[query_idx, :]) / self.n_classifiers
        pct_agree = np.sum(self.processed_dsel, axis=1) / self.n_classifiers

        return pct_agree

    def compute_meta_features(self, scores, idx_neighbors, idx_neighbors_op):
        """Compute the five sets of meta-features used in the META-des framework. Returns
        the meta-features vector :math:`v_{i,j}`.

        Parameters
        ----------
        scores : array of shape = [n_samples, n_classifiers, n_classes]
                 scores (posterior probabilities) obtained by the base classifiers for each sample to extract the meta
                 features.

        idx_neighbors : array of shape = [n_samples, self.K]
                        indices of K-nearest neighbors for each example.

        idx_neighbors_op : array of shape = [n_samples, self.Kp]
                           Indices of the most similar output profiles for each example.

        Returns
        -------
        meta_feature_vectors : array of shape [n_query x n_classifiers, n_meta_features]
                               The five sets of meta-features estimated for each pair (base classifier, example).

        """

        idx_neighbors = np.atleast_2d(idx_neighbors)
        idx_neighbors_op = np.atleast_2d(idx_neighbors_op)

        f1_all_classifiers = self.processed_dsel[idx_neighbors, :].swapaxes(1, 2).reshape(-1, self.k)

        f2_all_classifiers = self.dsel_scores[idx_neighbors, :, self.DSEL_target[idx_neighbors]].swapaxes(1, 2)
        f2_all_classifiers = f2_all_classifiers.reshape(-1, self.k)

        f3_all_classifiers = np.mean(self.processed_dsel[idx_neighbors, :], axis=1).reshape(-1, 1)

        f4_all_classifiers = self.processed_dsel[idx_neighbors_op, :].swapaxes(1, 2).reshape(-1, self.Kp)

        f5_all_classifiers = np.max(scores, axis=2).reshape(-1, 1)
        meta_feature_vectors = np.hstack((f1_all_classifiers, f2_all_classifiers, f3_all_classifiers,
                                          f4_all_classifiers, f5_all_classifiers))

        return meta_feature_vectors

    def _generate_meta_training_set(self):
        """Routine to generate the meta-training dataset that is further used to train the meta-classifier (Lambda).

        In this procedure we use a leave-one-out scheme in which
        each sample in DSEL is used as reference to generate the meta-features and all others are used to estimate the
        region of competence.

        The first step is to apply the sample selection mechanism in order to decide whether  or not the corresponding
        sample should be used for meta-training process. Then, for each base classifier, five sets of meta-features
        are calculated and added to the meta-training dataset.

        """
        # first compute the agreement of each sample for the sample selection mechanism
        agreement = self._sample_selection_agreement()
        indices_selected = np.hstack( (np.where(self.Hc > agreement)[0], np.where(agreement > (1 - self.Hc))[0]))
        indices_selected = np.unique(indices_selected)
        # Get the region of competence using the feature space and the decision space. Use K + 1 to later remove itself
        # from the set.
        _, idx_neighbors = self._get_region_competence(self.DSEL_data[indices_selected, :], self.k + 1)
        _, idx_neighbors_op = self._get_similar_out_profiles(self.dsel_scores[indices_selected], self.Kp + 1)
        # Remove the first neighbor (itself)
        idx_neighbors = idx_neighbors[:, 1:]
        idx_neighbors_op = idx_neighbors_op[:, 1:]

        # Get the scores for the samples that the meta features are being extracted
        scores = self.dsel_scores[indices_selected, :, :]

        # Extract the meta-feature vectors for each base classifier. vector and target must both be numpy arrays
        meta_feature_vector = self.compute_meta_features(scores, idx_neighbors, idx_neighbors_op)
        meta_feature_target = self.processed_dsel[indices_selected, :].reshape(-1,)
        meta_feature_target.astype(np.int)

        return meta_feature_vector, meta_feature_target

    def _fit_meta_classifier(self, X_meta, y_meta):
        """Train the meta-classifier :math:`\\lambda`, using the meta-training dataset.

        Parameters
        ----------
        X_meta : array of shape = [n_meta_examples, n_meta_features]
                 The meta-training examples.

        y_meta : array of shape = [n_meta_examples]
                 Class labels of each example in X_test. 1 whether the base classifier made the correct prediction,
                 otherwise 0.

        """
        if isinstance(self.meta_classifier, MultinomialNB):
            # Digitize the data (Same implementation we have on PRTools)
            X_meta = np.digitize(X_meta, np.linspace(0.1, 1, 10))

        self.meta_classifier.fit(X_meta, y_meta)

    def _get_similar_out_profiles(self, probabilities, kp=None):
        """Get the most similar output profiles of the query sample.

        Parameters
        ----------
        probabilities : array of shape = [n_samples, n_classifiers, n_classes]
                        Probabilities estimates by each each base classifier for each sample.

        kp : int
             The number of output profiles (most similar) to be selected.

        Returns
        -------
        dists : list of shape = [n_samples, k]
                The distances between the query and each sample in the region of competence. The vector is ordered
                in an ascending fashion.

        idx : list of shape = [n_samples, k]
              Indices of the instances belonging to the region of competence of the given query sample.
        """
        if kp is None:
            kp = self.Kp

        if self.n_classes == 2:
            # Get only the scores for one class since they are complementary
            query_op = probabilities[:, :, 0]
        else:
            query_op = probabilities.reshape((probabilities.shape[0], self.n_classifiers * self.n_classes))

        dists, idx = self.op_knn.kneighbors(query_op, n_neighbors=kp, return_distance=True)
        return dists, idx

    def select(self, competences):
        """Selects the base classifiers that obtained a competence level higher than the predefined
        threshold defined in self.selection_threshold.

        Parameters
        ----------
        competences : array of shape = [n_samples, n_classifiers]
                      The competence level estimated for each base classifier and test example.

        Returns
        -------
        selected_classifiers : array of shape = [n_samples, n_classifiers]
                               Boolean matrix containing True if the base classifier is select, False otherwise.

        """
        if competences.ndim < 2:
            competences = competences.reshape(1, -1)

        selected_classifiers = (competences > self.selection_threshold)
        # For the rows that are all False (i.e., no base classifier was selected, select all classifiers (all True)
        selected_classifiers[~np.any(selected_classifiers, axis=1), :] = True

        return selected_classifiers

    def estimate_competence_from_proba(self, query, probabilities):
        """Estimate the competence of each base classifier :math:`c_i`
        the classification of the query sample. This method received an array with the pre-calculated probability
        estimates for each query.

        First, the meta-features of each base classifier :math:`c_i` for the classification of the
        query sample are estimated. These meta-features are passed down to the meta-classifier :math:`\\lambda`
        for the competence level estimation.

        Parameters
        ----------
        query : array of shape = [n_samples, n_features]
                The test examples.

        probabilities : array of shape = [n_samples, n_classifiers, n_classes]
                        Probabilities estimates obtained by each each base classifier for each query sample.

        Returns
        -------
        competences : array of shape = [n_samples, n_classifiers]
                      The competence level estimated for each base classifier and test example.
        """
        _, idx_neighbors = self._get_region_competence(query)
        _, idx_neighbors_op = self._get_similar_out_profiles(probabilities)
        meta_feature_vectors = self.compute_meta_features(probabilities, idx_neighbors, idx_neighbors_op)

        # Digitize the data if a Multinomial NB is used as the meta-classifier
        if isinstance(self.meta_classifier, MultinomialNB):

            meta_feature_vectors = np.digitize(meta_feature_vectors, np.linspace(0.1, 1, 10))

        # Get the probability for class 1 (Competent)
        competences = self.meta_classifier.predict_proba(meta_feature_vectors)[:, 1]

        # Reshape the array from 1D [n_samples x n_classifiers] to 2D [n_samples, n_classifiers]
        competences = competences.reshape(-1, self.n_classifiers)
        if self.DFP:
            competences = competences * self.DFP_mask

        return competences

    @staticmethod
    def _check_input_parameters(Hc, gamma, meta_classifier):
        """Check if the parameters passed as argument are correct.

        - Hc should be higher than 0.5

        - selection_threshold should be higher than 0.5

        - The meta-classifier must implement the predict_proba function or be equal to None (Naive Bayes is
        used in its place).
        ----------
        """
        if not isinstance(Hc, (float, int)):
            raise ValueError('Parameter Hc should be either a number. Currently Hc = {}'.format(type(Hc)))
        if Hc < 0.5:
            raise ValueError('Parameter Hc should be higher than 0.5. Currently Hc = {}'.format(Hc))
        if not isinstance(gamma, float):
            raise ValueError('Parameter Hc should be either a float. Currently Hc = {}'.format(type(Hc)))
        if gamma < 0.5:
            raise ValueError('Parameter selection_threshold should be higher than 0.5. '
                             'Currently selection_threshold = {}'.format(gamma))
        if meta_classifier is not None and "predict_proba" not in dir(meta_classifier):
            raise ValueError("The meta-classifier should output probability estimates")


