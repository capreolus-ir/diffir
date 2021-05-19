import pytrec_eval
from diffir.measure import Measure


class QrelMeasure(Measure):
    module_name = "qrel"

    def _query_differences(self, run1, run2, *args, **kwargs):
        """
        :param run1: TREC run. Has the format {qid: {docid: score}, ...}
        :param run2: Same as above
        :param args:
        :param kwargs: Expects a 'dataset' parameter. This is an instance of ir-datasets
        :return: A list of qids that differ the most in the metric
        """
        assert "dataset" in kwargs, "Dataset object not supplied for qrel measure"
        dataset = kwargs["dataset"]
        assert dataset.has_qrels(), "Dataset object does not have the qrels files"
        overlapping_keys = set(run1.keys()).intersection(set(run2.keys()))
        run1 = {qid: doc_id_to_score for qid, doc_id_to_score in run1.items() if qid in overlapping_keys}
        run2 = {qid: doc_id_to_score for qid, doc_id_to_score in run2.items() if qid in overlapping_keys}

        qrels = dataset.qrels_dict()
        metric = self.metric
        topk = self.topk
        evaluator = pytrec_eval.RelevanceEvaluator(qrels, {metric})
        eval_run_1 = evaluator.evaluate(run1)
        eval_run_2 = evaluator.evaluate(run2)
        query_ids = eval_run_1.keys() & eval_run_2.keys()
        query_ids = sorted(query_ids, key=lambda x: abs(eval_run_1[x][metric] - eval_run_2[x][metric]), reverse=True)
        query_ids = query_ids[:topk]
        id2diff = {x: abs(eval_run_1[x][metric] - eval_run_2[x][metric]) for x in query_ids}
        return query_ids, id2diff, metric
