import json
from intervaltree import IntervalTree
from nltk import word_tokenize
from nltk.corpus import stopwords
from profane import ModuleBase, Dependency, ConfigOption
import ir_datasets
from . import Weight


_logger = ir_datasets.log.easy()


@Weight.register
class CustomWeight(Weight):
    module_name = "custom"
    config_spec = [
        # TODO: is there a better way to handle these args? There's strlist, but that's ugly for file paths.
        # Maybe we could infer them if they are named with the run prefix? Can we get the run paths here?
        ConfigOption(key="weights_1", default_value=None, description="TODO"),
        ConfigOption(key="norm_1", default_value="minmax", description="TODO"),
        ConfigOption(key="weights_2", default_value=None, description="TODO"),
        ConfigOption(key="norm_2", default_value="minmax", description="TODO"),
    ]

    def build(self):
        self._cache = {}
        for run_idx in [0, 1]:
            weights_file, norm = self.config[f"weights_{run_idx+1}"], self.config[f"norm_{run_idx+1}"]
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
