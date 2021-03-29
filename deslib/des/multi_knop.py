# coding=utf-8

# Author: Rafael Menelau Oliveira e Cruz <rafaelmenelau@gmail.com>
#
# License: BSD 3 clause

import numpy as np

from deslib.des.base import BaseDS
from deslib.des.knop import KNOP
from sklearn.utils.validation import (check_X_y, check_is_fitted, check_array,
                                      check_random_state)


class MultiKNOP(KNOP):
    def __init__(self, pool_classifiers=None, k=7, DFP=False, with_IH=False,
                 safe_k=None, IH_rate=0.30, random_state=None, voting='hard',
                 knn_classifier='knn', knne=False, DSEL_perc=0.5, n_jobs=-1):

        super(KNOP, self).__init__(None, k,
                                   DFP=DFP,
                                   with_IH=with_IH,
                                   safe_k=safe_k,
                                   IH_rate=IH_rate,
                                   needs_proba=True,
                                   random_state=random_state,
                                   knn_classifier=knn_classifier,
                                   knne=knne,
                                   DSEL_perc=DSEL_perc,
                                   n_jobs=n_jobs)
        self.ds_classifiers = []
        for i in range(len(pool_classifiers)):
            knop = KNOP(pool_classifiers[i], k,
                        DFP=DFP,
                        with_IH=with_IH,
                        safe_k=safe_k,
                        IH_rate=IH_rate,
                        random_state=random_state,
                        knn_classifier=knn_classifier,
                        knne=knne,
                        DSEL_perc=DSEL_perc,
                        n_jobs=n_jobs)
            self.ds_classifiers.append(knop)


    """Multi k-Nearest Output Profiles (MultiKNOP).
    """
    def fit(self, X, y):
        """Train the DS model by setting the KNN algorithm and
        pre-process the information required to apply the DS
        methods. In this case, the scores of the base classifiers for
        the dynamic selection dataset (DSEL) are pre-calculated to
        transform each sample in DSEL into an output profile.

        Parameters
        ----------
        X : array of shape n_datasets, n_samples, n_features)
            Data used to fit the model.

        y : array of shape (n_datasets, n_samples)
            class labels of each example in X.

        Returns
        -------
        self
        """
        if len(X) == 1 or len(y) == 1:
            raise ValueError("Error. MultiKNOP  does not accept one dataset!")
        """
        for predict()...
        for i in range(len(y)-1):
            if np.array_equal(y[0],y[i+1]):
                raise ValueError(
                    "Error. All datasets queries must match exactly!")
        """

        datasets_dsel_scores = []
        datasets_DSEL_processed_ = []
        # Process each dataset
        for i in range(len(X)):
            self.ds_classifiers[i].fit(X[i], y[i])
            if self.ds_classifiers[i].n_classes_ == 1:
                raise ValueError(
                    "Error. MultiKNOP  does not accept one class datasets!")
            self.ds_classifiers[i]._check_predict_proba()
            self.ds_classifiers[i].dsel_scores_ = \
                self.ds_classifiers[i]._preprocess_dsel_scores()
            datasets_dsel_scores.append(self.ds_classifiers[i].dsel_scores_)
            datasets_DSEL_processed_.append(
                self.ds_classifiers[i].DSEL_processed_)
        
        self.dsel_scores_ = np.concatenate(datasets_dsel_scores, axis=1)
        self.DSEL_processed_ = np.concatenate(datasets_DSEL_processed_, axis=1)
        self.n_classifiers_ = self.dsel_scores_.shape[1]
        
        # Reassignment
        self.DSEL_target_ = self.ds_classifiers[0].DSEL_target_
        self.n_samples_ = self.ds_classifiers[0].n_samples_
        self.n_classes_ = self.ds_classifiers[0].n_classes_
        self.knn_class_ = self.ds_classifiers[0].knn_class_
        self.k_ = self.ds_classifiers[0].k_
        self.classes_ = self.ds_classifiers[0].classes_
        
        # Reshape DSEL_scores as a 2-D array for nearest neighbor calculations
        dsel_output_profiles = self.dsel_scores_.reshape(self.n_samples_,
                                                         self.n_classifiers_ *
                                                         self.n_classes_)

        self._fit_OP(dsel_output_profiles, self.DSEL_target_, self.k_)

        return self

    def _fit_OP(self, X_op, y_op, k):
        """ Fit the set of output profiles.

        Parameters
        ----------
        X_op : array of shape (n_samples, n_features)
            Output profiles of the training data. n_features is equals
            to (n_classifiers x n_classes).

        y_op : array of shape (n_samples)
               Class labels of each sample in X_op.

        k : int
            Number of output profiles used in the region of competence
            estimation.

        """
        self.op_knn_ = self.knn_class_(k)

        if self.n_classes_ == 2:
            # Get only the scores for one class since they are complementary
            X_temp = X_op[:, ::2]
            self.op_knn_.fit(X_temp, y_op)
        else:
            self.op_knn_.fit(X_op, y_op)

    def _get_similar_out_profiles(self, probabilities):
        """Get the most similar output profiles of the query sample.

        Parameters
        ----------
        probabilities : array of shape (n_samples, n_classifiers, n_classes)
                        predictions of each base classifier for all samples.

        Returns
        -------
        dists : list of shape = [n_samples, k]
                The distances between the query and each sample in the region
                of competence. The vector is ordered in an ascending fashion.

        idx : list of shape = [n_samples, k]
            Indices of the instances belonging to the region of competence of
            the given query sample.
        """

        if self.n_classes_ == 2:
            # Get only the scores for one class since they are complementary
            query_op = probabilities[:, :, 0]
        else:
            query_op = probabilities.reshape((probabilities.shape[0],
                                              self.n_classifiers_ *
                                              self.n_classes_))

        dists, idx = self.op_knn_.kneighbors(query_op, n_neighbors=self.k_,
                                             return_distance=True)
        return dists, np.atleast_2d(idx)

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
        # Check if the DS model was trained
        #check_is_fitted(self,
        #                ["DSEL_processed_", "DSEL_data_", "DSEL_target_"])

        # Check if X is a valid input
        #for i in range(len(X)):
        #    X[i] = check_array(X[i])
        #    self._check_num_features(X[i])

        n_samples = X[0].shape[0]
        predicted_labels = np.empty(n_samples, dtype=np.intp)

        base_probabilities = []
        for i in range(len(X)):
            base_probabilities.append(
                self.ds_classifiers[i]._predict_proba_base(X[i]))
        base_probabilities = np.concatenate(base_probabilities,axis=1)
        base_predictions = base_probabilities.argmax(axis=2)

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

            X_DS = X[0][ind_disagreement, :]

            # Then, we estimate the nearest neighbors for all samples that
            # we need to call DS routines
            distances, neighbors = None, None
            #distances, neighbors = self._get_region_competence(X_DS)

            # IH was not considered. So all samples with disagreement are
            # passed down to the DS algorithm
            ind_ds_classifier = np.arange(ind_disagreement.size)

            # At this stage the samples which all base classifiers agrees or
            # that are associated with low hardness were already classified.
            # The remaining samples are now passed down to the DS techniques
            # for classification.

            #  First check whether there are still samples to be classified.
            if ind_ds_classifier.size:

                DFP_mask = np.ones(
                    (ind_ds_classifier.size, self.n_classifiers_))

                # Get the real indices_ of the samples that will be classified
                # using a DS algorithm.
                ind_ds_original_matrix = ind_disagreement[ind_ds_classifier]

                if self.needs_proba or self.voting == 'soft':
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
