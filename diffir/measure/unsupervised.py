from profane import ModuleBase, Dependency, ConfigOption
from diffir.measure import Measure
from scipy import stats
import numpy as np


@Measure.register
class TopkMeasure(Measure):
    module_name = "topk"
    config_spec = [
        ConfigOption(key="topk", default_value=3, description="TODO"),
        ConfigOption(key="metric", default_value="weightedtau", description="Metric to measure the rank correaltion"),
    ]

    def tauap(self, x, y, decreasing=True):
        """
        AP Rank correalation Coefficient
        :param x: a list of scores
        :param y: another list of scores for comparision
        :param decreasing:
        :return:
            float: the correlation coefficient
        """
        rx = stats.rankdata(x)
        ry = stats.rankdata(y)
        n = len(rx)

        numerator = 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                sx = np.sign(x[i] - x[j])
                sy = np.sign(y[i] - y[j])
                if sx == sy:
                    numerator += 1 / (n - min(ry[i], ry[j]))
        return (2 * numerator / (n - 1)) - 1

    def tauap_fast(self, x, y):
        """
        AP Ranking Correlation using enhanced merge sort
        :param x:
        :param y:
        :return:
        """
        rx = stats.rankdata(x)
        ry = stats.rankdata(y)
        n = len(ry)
        if n == 1:
            return 1
        ordered_idx = sorted(list(range(n)), key=lambda i: rx[i])
        ry_ordered_by_rx = [(ry[idx], i) for i, idx in enumerate(ordered_idx)]

        def merge_sort(arr):
            if len(arr) <= 1:
                return 0
            mid = len(arr) // 2
            L = arr[:mid]
            R = arr[mid:]
            tauAP = 0
            tauAP += merge_sort(L)
            tauAP += merge_sort(R)
            i = j = k = 0
            c = 0
            while i < len(L) and j < len(R):
                if L[i][0] <= R[j][0]:
                    arr[k] = L[i]
                    if (n - L[i][1]) > 1:
                        tauAP += j / (n - L[i][1] - 1)
                    i += 1
                else:
                    arr[k] = R[j]
                    j += 1
                k += 1
            while i < len(L):
                arr[k] = L[i]
                if (n - L[i][1]) > 1:
                    tauAP += j / (n - L[i][1] - 1)
                i += 1
                k += 1
            while j < len(R):
                arr[k] = R[j]
                j += 1
                k += 1
            return tauAP

        res = (2 - 2 * merge_sort(ry_ordered_by_rx) / (n - 1)) - 1
        return res

    def _query_differences(self, run1, run2, *args, **kwargs):
        """
        :param run1: TREC run. Has the format {qid: {docid: score}, ...}
        :param run2: Same as above
        :param args:
        :param kwargs:
        :return: The union of top k qids in both runs, sorted by the order in which the queries appear in run 1
        ^ This is because run 1 appears on the left hand side in the web ui
        """
        topk = self.config["topk"]
        metric = self.config["metric"]
        qids = run1.keys() & run2.keys()
        if not qids:
            raise ValueError("run1 and run2 have no shared qids")

        id2measure = {}
        for qid in qids:
            from collections import defaultdict

            doc_score_1 = defaultdict(lambda: 0, run1[qid])
            doc_score_2 = defaultdict(lambda: 0, run2[qid])
            doc_ids_1 = doc_score_1.keys()
            doc_ids_2 = doc_score_2.keys()
            doc_ids_union = set(doc_ids_1).union(set(doc_ids_2))
            doc_ids_union = sorted(list(doc_ids_union), key=lambda id: (doc_score_1[id] + doc_score_2[id]), reverse=True)
            union_score1 = [doc_score_1[doc_id] for doc_id in doc_ids_union]
            union_score2 = [doc_score_2[doc_id] for doc_id in doc_ids_union]
            if metric == "weightedtau":
                tau, p_value = stats.weightedtau(union_score1, union_score2)
            elif metric == "tauap":
                tau = self.tauap_fast(union_score1, union_score2)
            else:
                raise ValueError("Metric {} not supported for the measure {}".format(self.config["metric"], self.module_name))
            id2measure[qid] = tau
        qids = sorted(qids, key=lambda x: id2measure[x])
        return qids[:topk]
