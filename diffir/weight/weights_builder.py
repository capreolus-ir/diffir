from .custom import CustomWeight
from .unsupervised import ExactMatchWeight


class WeightBuilder:
    def __init__(self, weights_1, weights_2, norm_1="minmax", norm_2="minmax"):
        self.weights = [self.initialize_weights(weights_1, norm_1), self.initialize_weights(weights_2, norm_2)]

    @staticmethod
    def initialize_weights(weights_file, norm):
        if weights_file is None:
            return ExactMatchWeight()
        else:
            return CustomWeight(weights_file, norm)

    def score_document_regions(self, query, doc, run_idx):
        return self.weights[run_idx].score_document_regions(query, doc)
