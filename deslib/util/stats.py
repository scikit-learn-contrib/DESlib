import numpy as np


class Stats():
    def __init__(self):
        self.agree_ind = []
        self.disagree_ind = []
        self.true_labels = []
        self.bases_labels = []
        self.predicted_labels = []
        self.agree_labels = []
        self.competences = []
        self.log_fname = "log.txt"

    def log_stats(self):
        bln_mat = np.equal(self.true_labels,self.predicted_labels)
        self.wrong_true_labels = self.true_labels[~bln_mat]
        self.wrong_bases_labels = self.bases_labels[~bln_mat]
        self.wrong_predicted_labels = self.predicted_labels[~bln_mat]
        self.wrong_competences = self.competences[~bln_mat]

        self.n_queries = len(self.wrong_true_labels)
        self.n_bases = len(self.bases_labels[0])
        self.n_disagree = len(self.disagree_ind)

        with open(self.log_fname,'w') as f:
            for line in self._get_all_lines():
                f.write(str(line))
                f.write("\n")

    def _get_all_lines(self):
        lines = []
        lines.extend(self._get_general_lines())
        lines.extend(self._get_agree_lines())
        lines.extend(self._get_n_right_clf_lines())
        if self.n_disagree > 0:
            lines.extend(self._get_disagree_lines())
            lines.extend(self._get_competences_lines())
            lines.extend(self._get_competences_reliability_lines())
        return lines

    def _get_general_lines(self):
        lines = [
            "Queries:",
            self.n_queries,
        ]

        return lines

    def _get_agree_lines(self):
        n_agree = len(self.agree_ind)
        agree_dis = self._get_distribution(ind=self.agree_ind)
        agree_score = self._get_score(
            self.agree_ind, self.true_labels, self.predicted_labels)
        predicted_dis = self._get_distribution()
        
        lines = [
            "--- Agreements",
            "Instances, ratio on queries:",
            n_agree,
            round(n_agree / self.n_queries, 3),
            "Distribution, ratio on predictions:",
            agree_dis,
            np.round(agree_dis / predicted_dis, 3),
            "Score, ratio on agreements:",
            agree_score,
            round(agree_score / n_agree, 3),
        ]

        return lines

    def _get_n_right_clf_lines(self):
        n_right_clf_by_query, n_right_clf_ind = \
            self._get_n_right_clf_stats(self.wrong_true_labels,self.wrong_bases_labels)
        n_right_clf_dis = self._get_distribution(n_right_clf_by_query)
        scores = [self._get_score(
            n_right_clf_ind[i],
            self.wrong_true_labels,
            self.wrong_predicted_labels) \
            for i in range(len(n_right_clf_dis))]

        lines = [
            "--- Right classifiers:",
            "Distribution, ratio on queries:",
            n_right_clf_dis,
            np.round(n_right_clf_dis / self.n_queries, 3),
            "Scores, ratio on N right clf",
            scores,
            np.round(scores / n_right_clf_dis, 3),
        ]

        return lines

    def _get_disagree_lines(self):
        disagree_score = self._get_score(
            self.disagree_ind, self.true_labels, self.predicted_labels)
        lines = [
            "--- Disagreements",
            "Instances, ratio on queries:",
            self.n_disagree,
            round(self.n_disagree / self.n_queries, 3),
            "Score, ratio on disagreements:",
            disagree_score,
            round(disagree_score / self.n_disagree, 3),
        ]

        return lines

    def _get_competences_lines(self):
        mean, mean_by_clf, var, var_by_clf, n_even_max = \
            self._get_competences_stats(self.wrong_competences)

        lines = [
            "--- Competences",
            "Mean:",
            round(mean, 3),
            "Mean by classifier:",
            np.round(mean_by_clf, 3),
            "Var:",
            round(var, 3),
            "Var by classifier:",
            np.round(var_by_clf, 3),
            "Even max competences times, \nratio on disagreements:",
            n_even_max,
            round(n_even_max / self.n_disagree, 3),
        ]

        return lines

    def _get_competences_reliability_lines(self):
        true_labels_by_base = np.tile(self.true_labels,(self.n_bases,1))
        correct_bases_bln_array = np.equal(
            self.bases_labels.T,true_labels_by_base)
        n_queries = self.bases_labels.shape[0]
        n_correct_labels_by_base = np.sum(correct_bases_bln_array,axis=1)
        acc_by_base = np.round(n_correct_labels_by_base/n_queries,3)

        comp = self.competences/self.k
        n_incorrect_labels_by_base = \
            len(self.predicted_labels)-n_correct_labels_by_base

        correct_comp = comp.T*correct_bases_bln_array
        correct_comp_by_base = np.sum(correct_comp,axis=1)
        correct_comp_by_base /= n_correct_labels_by_base
        mean_correct_comp_by_base = np.round(correct_comp_by_base,3)
        mean = mean_correct_comp_by_base
        mean = np.repeat(mean,n_queries).reshape(self.n_bases,-1)
        correct_comp_by_base = np.sum((correct_comp-mean)**2,axis=1)
        correct_comp_by_base /= n_correct_labels_by_base
        std_correct_comp_by_base = np.round(np.sqrt(correct_comp_by_base),3)

        incorrect_comp = comp.T*~correct_bases_bln_array
        incorrect_comp_by_base = np.sum(incorrect_comp,axis=1)
        incorrect_comp_by_base /= n_incorrect_labels_by_base
        mean_incorrect_comp_by_base = np.round(incorrect_comp_by_base,3)
        mean = mean_incorrect_comp_by_base
        mean = np.repeat(mean,n_queries).reshape(self.n_bases,-1)
        incorrect_comp_by_base = np.sum((incorrect_comp-mean)**2,axis=1)
        incorrect_comp_by_base /= n_incorrect_labels_by_base
        std_incorrect_comp_by_base = np.round(
            np.sqrt(incorrect_comp_by_base),3)

        lines = [
            "--- Competence reliability",
            "Acc:",
            round(np.mean(acc_by_base),3),
            "(by base):",
            acc_by_base,
            "Competence mean & std when well clasified:",
            round(np.mean(mean_correct_comp_by_base),3),
            round(np.mean(std_correct_comp_by_base),3),
            "(by base):",
            mean_correct_comp_by_base,
            std_correct_comp_by_base,
            "Competence mean & std when not well clasified:",
            round(np.mean(mean_incorrect_comp_by_base),3),
            round(np.mean(std_incorrect_comp_by_base),3),
            "(by base):",
            mean_incorrect_comp_by_base,
            std_incorrect_comp_by_base,
        ]

        return lines

    def _get_distribution(self, labels=None, ind=None):
        labels = self.predicted_labels if labels is None else labels
        max_label = max(labels)
        if ind is not None: labels = labels[ind]
        unique_labels, unique_counts = np.unique(labels, return_counts=True)
        distribution = np.full(max_label+1,0)
        for i,l in enumerate(unique_labels):
            distribution[l] = unique_counts[i]
        return distribution

    def _get_n_right_clf_stats(self,true_labels,bases_labels):
        n_right_clf_by_query = []
        n_right_clf_ind = [[] for i in range(self.n_bases + 1)]

        for i,label in enumerate(true_labels):
            row = bases_labels[i]
            n_right_clf = np.count_nonzero(row == label)
            n_right_clf_by_query.append(n_right_clf)
            n_right_clf_ind[n_right_clf].append(i)

        return n_right_clf_by_query, n_right_clf_ind

    def _get_competences_stats(self, competences):
        competences = competences/self.k
        mean = np.mean(competences)
        var = np.var(competences)
        mean_by_clf = np.mean(competences, axis=0)
        var_by_clf = np.var(competences, axis=0)
        n_even_max = 0

        for c in competences:
            max_ = c[np.argmax(c)]
            n_max = np.count_nonzero(c == max_)
            if n_max > 1: n_even_max += 1

        return mean, mean_by_clf, var, var_by_clf, n_even_max

    def _get_score(self, ind, true_labels, predicted_labels):
        matches = np.equal(true_labels[ind], predicted_labels[ind])
        score = np.sum(matches)
        return score


class MultiStats(Stats):
    def __init__(self):
        super().__init__()
        self.n_datasets = 1

    def _get_all_lines(self):
        lines = super()._get_all_lines()
        lines.extend(self._get_multistats_lines())
        return lines

    def _get_multistats_lines(self):
        lines = [
            "--- Multidatasets",
            self.n_datasets
        ]

        return lines
