from ir_measures import iter_calc, parse_measure
import sys
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
        try:
            metric = parse_measure(self.metric)
        except NameError:
            print("Unknown measure: {}. Please provide a measure supported by https://ir-measur.es/".format(self.metric))
            sys.exit(1)

        topk = self.topk
        eval_run_1 = self.convert_to_nested_dict(iter_calc([metric], qrels, run1))
        eval_run_2 = self.convert_to_nested_dict(iter_calc([metric], qrels, run2))

        query_ids = eval_run_1.keys() & eval_run_2.keys()
        query_ids = sorted(query_ids, key=lambda x: abs(eval_run_1[x][metric] - eval_run_2[x][metric]), reverse=True)
        query_ids = query_ids[:topk]
        id2diff = {x: abs(eval_run_1[x][metric] - eval_run_2[x][metric]) for x in query_ids}
        id2qrelscores = {x: [eval_run_1[x][metric], eval_run_2[x][metric]] for x in query_ids}
        return query_ids, id2diff, self.metric, id2qrelscores

    def convert_to_nested_dict(self, ir_measures_iterator):
        """
        Util method to convert the results from ir_measures.iter_calc to a dict.
        TODO: We can probably refactor so that this method won't be needed
        """
        eval_dict = {}

        for x in ir_measures_iterator:
            # TODO: This assumes that there would be only one measure/metric to handle.
            eval_dict[x.query_id] = {x.measure: x.value}

        return eval_dict
