import json
import ir_datasets
from . import Weight

_logger = ir_datasets.log.easy()


class CustomWeight(Weight):
    module_name = "custom"

    def __init__(self, weights_file, norm="minmax"):
        """
        Customed weights file from the ranking model
        :param weights_file: weights_file extracted from the ranking model
        :param norm: approach to normalize weight        
        """
        self.weights_file = weights_file
        self.norm = norm
        self._cache = {}
        self.build()

    def build(self):
        with open(self.weights_file, "rt") as f:
            # In case of multiple folds, each line in the file would be a valid json representing weights for that fold
            combined_weights = {}

            for line in f:
                fold_weights = json.loads(line)
                combined_weights.update(fold_weights)
            self._cache = combined_weights

        if self.norm == "none":
            pass
        elif self.norm == "minmax":
            # normalize by min/max weights *within each query
            for qid, docs in self._cache.items():
                min_weight = min(w[2] for f in docs.values() for wts in f.values() for w in wts)
                max_weight = max(w[2] for f in docs.values() for wts in f.values() for w in wts)
                for fields in docs.values():
                    for weights in fields.values():
                        for weight in weights:
                            weight[2] = (weight[2] - min_weight) / (max_weight - min_weight)
        else:
            raise ValueError(f"unknown norm {self.norm}")

    def score_document_regions(self, query, doc):
        result = self._cache.get(query.query_id, {}).get(doc.doc_id, {})
        return result
