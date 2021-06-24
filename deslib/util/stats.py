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
        self.n_queries = len(self.true_labels)
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
        agree_score = self._get_score(self.agree_ind)
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
            self._get_n_right_clf_stats()
        n_right_clf_dis = self._get_distribution(n_right_clf_by_query)
        scores = [self._get_score(n_right_clf_ind[i]) \
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
        disagree_score = self._get_score(self.disagree_ind)

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
        competences_mean, competences_mean_by_clf, n_even_max_competence = \
            self._get_competences_stats()

        lines = [
            "--- Competences",
            "Mean:",
            round(competences_mean, 3),
            "Mean by classifier:",
            np.round(competences_mean_by_clf, 3),
            "Even max competences times, \nratio on disagreements:",
            n_even_max_competence,
            round(n_even_max_competence / self.n_disagree, 3),
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

    def _get_n_right_clf_stats(self):
        n_right_clf_by_query = []
        n_right_clf_ind = [[] for i in range(self.n_bases + 1)]

        for i,label in enumerate(self.true_labels):
            row = self.bases_labels[i]
            n_right_clf = np.count_nonzero(row == label)
            n_right_clf_by_query.append(n_right_clf)
            n_right_clf_ind[n_right_clf].append(i)

        return n_right_clf_by_query, n_right_clf_ind

    def _get_competences_stats(self):
        competences_mean = np.mean(self.competences)
        competences_mean_by_clf = np.mean(self.competences, axis=0)
        n_even_max_competence = 0

        for c in self.competences:
            max_ = c[np.argmax(c)]
            n_max = np.count_nonzero(c == max_)
            if n_max > 1: n_even_max_competence += 1

        return competences_mean, competences_mean_by_clf, n_even_max_competence

    def _get_score(self, ind):
        true_labels = self.true_labels[ind]
        labels = self.predicted_labels[ind]
        matches = np.equal(true_labels, labels)
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

    def _get_competences_lines(self):
        competences_mean, competences_mean_by_clf, n_even_max_competence = \
            self._get_competences_stats()

        means = competences_mean_by_clf.reshape(self.n_datasets, -1)
        competences_mean_by_dataset = np.mean(means, axis=1)

        lines = super()._get_competences_lines()
        lines.extend([
            "Mean by dataset:",
            np.round(competences_mean_by_dataset, 3),
        ])

        return lines

    def _get_multistats_lines(self):
        lines = [
            "--- Multidatasets",
            self.n_datasets
        ]

        return lines