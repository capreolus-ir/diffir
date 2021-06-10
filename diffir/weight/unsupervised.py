import re
import nltk
from intervaltree import IntervalTree
from nltk import word_tokenize
from nltk.corpus import stopwords
from . import Weight
import ahocorasick


class ExactMatchWeight(Weight):
    module_name = "exactmatch"

    def __init__(self, queryfield="", skip_stopwords=True):
        self.queryfield = queryfield
        self.skip_stopwords = skip_stopwords

    def fast_score_document_regions(self, query, doc):
        """
        Score document regions with Aho–Corasick_algorithm.
        :param query:
        :param doc:
        :param run_idx:
        :return:
        """
        try:
            nltk.data.find("tokenizers/punkt")
        except (LookupError, OSError):
            nltk.download("punkt")
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")
        result = {}
        stops = stopwords.words("english") if self.skip_stopwords else None
        query_tokens = set()
        for qfield_value in query:
            query_tokens.update(
                [
                    w.lower()
                    for w in word_tokenize(qfield_value)
                    if (w.isalpha() or w.isnumeric()) and (not stops or not w.lower() in stops)
                ]
            )
        if not hasattr(self, "A"):
            self.A = ahocorasick.Automaton()
        self.A.clear()
        for idx, token in enumerate(query_tokens):
            self.A.add_word(token, (idx, token))
        self.A.make_automaton()

        for dfield, dvalue in zip(doc._fields, doc):
            matches = [(end_idx - len(match) + 1, end_idx) for end_idx, (_, match) in self.A.iter(dvalue.lower())]
            result[dfield] = matches

        for field, values in list(result.items()):
            tree = IntervalTree()
            for start, stop in values:
                tree[start : stop + 1] = 1
            tree.merge_overlaps()
            result[field] = sorted([[i.begin, i.end, 1.0] for i in tree])
        return result

    def score_document_regions(self, query, doc, fast=False):
        if fast:
            return self.fast_score_document_regions(query, doc)

        try:
            nltk.data.find("tokenizers/punkt")
        except (LookupError, OSError):
            nltk.download("punkt")
        try:
            nltk.data.find("corpora/stopwords")
        except LookupError:
            nltk.download("stopwords")
        result = {}
        stops = stopwords.words("english") if self.skip_stopwords else None

        qfield_values = []
        specified_qfields = list(filter(None, self.queryfield))
        # Choose a query field to do the highlighting with
        if specified_qfields:
            for fname in specified_qfields:
                qfield_values.append(query._asdict()[fname])
        else:
            # Use the first field in the query that is not the id
            # ._asdict() is an OrderedDict, so this is deterministic
            for fname, fval in query._asdict().items():
                if fname != "query_id":
                    qfield_values = [fval]
                    break

        assert len(qfield_values)
        for qfield_value in qfield_values:
            for word in word_tokenize(qfield_value):
                word = word.lower()
                if not word.isalpha() and not word.isnumeric():
                    continue
                if stops and word.lower() in stops:
                    continue
                for dfield, dvalue in zip(doc._fields, doc):
                    if not isinstance(dvalue, str):
                        continue  # skip non-strings for now
                    if dfield not in result:
                        result[dfield] = []
                    for match in re.finditer("\\b" + re.escape(word) + "\\b", dvalue.lower()):
                        result[dfield].append([match.start(), match.end()])
        for field, values in list(result.items()):
            tree = IntervalTree()
            for start, stop in values:
                tree[start:stop] = 1
            tree.merge_overlaps()
            result[field] = sorted([[i.begin, i.end, 1.0] for i in tree])
        return result
