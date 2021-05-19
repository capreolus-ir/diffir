import json
import ir_datasets
from . import Weight
_logger = ir_datasets.log.easy()

class CustomWeight(Weight):
    module_name = "custom"
    def __init__(self, weights_1, weights_2, norm_1="minmax", norm_2="minmax"):
        '''
        Customed weights file from ranking models
        :param weights_1:
        :param weights_2:
        :param norm_1:
        :param norm_2:
        '''
        self.weights = [weights_1, weights_2]
        self.norms = [norm_1, norm_2]
        self.build()

    def build(self):
        self._cache = {}
        for run_idx in [0, 1]:
            weights_file, norm = self.weights[run_idx], self.norms[run_idx]
            if weights_file is None:
                _logger.warn(f"missing weights.weights_{run_idx + 1}")
                self._cache[run_idx] = {}
            else:
                with open(weights_file, "rt") as f:
                    # In case of multiple folds, each line in the file would be a valid json representing weights for that fold
                    combined_weights = {}

                    for line in f:
                        fold_weights = json.loads(line)
                        combined_weights.update(fold_weights)

                    self._cache[run_idx] = combined_weights

                if norm == "none":
                    pass
                elif norm == "minmax":
                    # normalize by min/max weights *within each query
                    for qid, docs in self._cache[run_idx].items():
                        min_weight = min(w[2] for f in docs.values() for wts in f.values() for w in wts)
                        max_weight = max(w[2] for f in docs.values() for wts in f.values() for w in wts)
                        for fields in docs.values():
                            for weights in fields.values():
                                for weight in weights:
                                    weight[2] = (weight[2] - min_weight) / (max_weight - min_weight)
                else:
                    raise ValueError(f"unknown norm {norm}")

    def score_document_regions(self, query, doc, run_idx):
        weights = self._cache[run_idx]
        result = weights.get(query.query_id, {}).get(doc.doc_id, {})
        return result
