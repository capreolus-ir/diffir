class Measure:
    def __init__(self, metric="ndcg_20", topk=5):
        """
        Measure construction
        :param metric: The metric used for selecting queries.
        :param topk: How many queries to retrieve
        """
        self.metric = metric
        self.topk = topk

    def query_differences(self, run1, run2, *args, **kwargs):
        """
        :param run1: the first run
        :param run2: the second run
        :param args:
        :param kwargs:
        :return:
        """
        if run1 and run2:
            return self._query_differences(run1, run2, *args, **kwargs)
        elif run1 and run2 is None:
            qids = sorted(list(run1.keys()))[: self.topk]
            id2diff = {qid: 0 for qid in qids}
            return qids, id2diff, "singlerun", None

    def _query_differences(self, run1, run2, *args, **kwargs):
        raise NotImplementedError
