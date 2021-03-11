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

    def log_stats(self):
        n_queries = len(self.true_labels)
        n_classes = len(np.unique(self.predicted_labels))
        n_bases = len(self.bases_labels[0])
        n_agree = len(self.agree_ind)
        n_disagree = len(self.disagree_ind)
        
        n_right_clf_by_query, n_right_clf_ind = \
            self._get_n_right_clf_stats(n_bases)

        predicted_dis = self._get_distribution()
        agree_dis = self._get_distribution(ind=self.agree_ind)
        n_right_clf_dis = self._get_distribution(n_right_clf_by_query)
        
        agree_score = self._get_score(self.agree_ind)
        disagree_score = self._get_score(self.disagree_ind)
        
        competences_mean, competences_mean_by_clf, n_even_max_competence = \
            self._get_competences_stats()

        lines = []
        lines.extend([
            "Queries:",
            n_queries,
            "Nb of right classifiers from 0 to "+str(n_bases)+":",
            n_right_clf_dis,
            "--- Agreements",
            "Instances, ratio on queries:",
            n_agree,
            round(n_agree / n_queries, 3),
            "Classes distribution, ratio on predictions:",
            agree_dis,
            np.round(agree_dis / predicted_dis, 3),
            "Score, ratio on agreements:",
            agree_score,
            round(agree_score / n_agree, 3),
        ])

        for i,n_right_clf in enumerate(n_right_clf_dis):
            score = self._get_score(n_right_clf_ind[i])
            lines.extend([
                "--- "+str(i)+" right classifiers",
                "Instances, ratio on queries:",
                n_right_clf_dis[i],
                round(n_right_clf / n_queries, 3),
                "Score, ratio on "+str(i)+" right clf:",
                score,
                round(score / n_right_clf_dis[i], 3),
            ])

        lines.extend([
            "--- Disagreements",
            "Instances, ratio on queries:",
            n_disagree,
            round(n_disagree / n_queries, 3),
            "Score, ratio on disagreements:",
            disagree_score,
            round(disagree_score / n_disagree, 3),
            "--- Competences",
            "Mean:",
            round(competences_mean, 3),
            "Mean by classifier:",
            np.round(competences_mean_by_clf, 3),
            "Even max competences times, ratio on disagreements:",
            n_even_max_competence,
            round(n_even_max_competence / n_disagree, 3),
        ])
        
        with open("log.txt",'w') as f:
            for line in lines:
                f.write(str(line))
                f.write("\n")

    def _get_distribution(self, labels=None, ind=None):
        labels = self.predicted_labels if labels is None else labels
        if ind is not None: labels = labels[ind]
        _, counts = np.unique(labels, return_counts=True)
        return counts

    def _get_n_right_clf_stats(self, n_bases):
        n_right_clf_by_query = []
        n_right_clf_ind = [[] for i in range(n_bases + 1)]

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
