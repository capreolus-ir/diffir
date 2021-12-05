import os
import numpy as np
import argparse
import json
from collections import defaultdict
from tqdm import tqdm
from intervaltree import IntervalTree, Interval
from mako.template import Template
from rich.console import Console
from rich.table import Table
from rich.prompt import Confirm
from rich.panel import Panel
import ir_datasets
from ir_measures import iter_calc
from ir_measures import P, nDCG
from diffir import QrelMeasure, TopkMeasure, WeightBuilder
from diffir.utils import load_trec_run
from typing import List, Dict, Callable

_logger = ir_datasets.log.easy()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("runfiles", nargs="+", help="run file(s) to display and compare")
    parser.add_argument("-c", "--cli", dest="cli", action="store_true", help="output to CLI (default)")
    parser.add_argument("-w", "--web", dest="web", action="store_true", help="output HTML file for WebUI")
    parser.add_argument("--dataset", dest="dataset", type=str, required=True, help="dataset identifier from ir_datasets")
    parser.add_argument(
        "--measure", dest="measure", type=str, default="qrel", help="measure for ranking difference (qrel, tauap, weightedtau)"
    )
    parser.add_argument(
        "--metric", dest="metric", type=str, default="nDCG@10", help="metric to report and used with qrel measure"
    )
    parser.add_argument("--topk", dest="topk", type=int, default=50, help="number of queries to compare")
    parser.add_argument("--weights_1", dest="weights_1", type=str, default=None, required=False)
    parser.add_argument("--weights_2", dest="weights_2", type=str, default=None, required=False)
    parser.add_argument("--qfield_to_use", dest="qfield", type=str, default="", required=False, help="Query field for exact matching")
    args = parser.parse_args()
    config = {
        "dataset": args.dataset,
        "measure": args.measure,
        "metric": args.metric,
        "topk": args.topk,
        "weight": {"weights_1": args.weights_1, "weights_2": args.weights_2},
        "qfield_to_use": args.qfield
    }
    if not (args.cli or args.web):
        args.cli = True  # default
    diff(args.runfiles, config, cli=args.cli, web=args.web)


def diff(runs, config, cli, web, print_html=True):
    for i, run in enumerate(runs):
        if config["weight"][f"weights_{i + 1}"] is None:
            if os.path.exists(run + ".diffir"):
                _logger.info("Found weight file at {}".format(run + ".diffir"))
                config["weight"][f"weights_{i + 1}"] = run + ".diffir"
            else:
                _logger.info("No weight file for {}. Fall to exact matching".format(run))
    _logger.info("Loading IR dataset")
    dataset = ir_datasets.load(config["dataset"])    
    config["qrels"] = dataset.qrels_dict()    
    query_dict = {}
    for query in dataset.queries_iter():
        query_dict[query.query_id] = query._asdict()
    config["queries"] = query_dict
    config["doc_lookup_fnc"] = lambda x : dataset.docs_store().get(x)._asdict()
    config["qrel_defs"] = dataset.qrels_defs()
    task = MainTask(**config)
    if cli:
        task.cli(runs)
    if web:
        html = task.web(runs)
        if print_html:
            print(html)
        return config, html

def diff_html(runs: List, queries: List[Dict], doc_lookup_fnc: Callable[[], Dict], qrels: Dict, dataset: str ="none", measure="qrel", metric="nDCG@10", topk: int =3, weight: Dict = {"weights_1": None, "weights_2": None}, qrel_defs: Dict = None, qfield_to_use: str = "", relevance_level: int  = 1, num_doc: int = 10):
    task = MainTask(queries, doc_lookup_fnc, qrels, dataset, measure, metric, topk, weight, qrel_defs, qfield_to_use, relevance_level, num_doc)
    return task.web(runs)

class MainTask:

    def __init__(self, queries: List[Dict], doc_lookup_fnc: Callable[[], Dict], qrels: Dict, dataset: str ="none", measure="qrel", metric="nDCG@5", topk: int =3, weight: Dict = {}, qrel_defs: Dict =None, qfield_to_use: str = "", relevance_level: int  = 1, num_doc: int = 10):
        self.dataset = dataset
        self.qrels = qrels        
        self.queries = queries
        self.doc_lookup_func = doc_lookup_fnc
        self.qfields = list(next(iter(self.queries.values())).keys())        
        self.qfield_to_use = qfield_to_use
        self.num_doc = num_doc
        self.relevance_level = relevance_level
        self.qrel_defs = qrel_defs
        if measure == "qrel":
            self.measure = QrelMeasure(metric, topk)
        elif measure in ["tauap", "weightedtau", "spearmanr", "pearsonrank", "kldiv"]:
            self.measure = TopkMeasure(measure, topk)
        else:
            raise ValueError("Measure {} is not supported".format(measure))
        #  query_fields=qfields
        self.weight = WeightBuilder(weight["weights_1"], weight["weights_2"])
    
    def get_relevant_docids(self, qrels):
        """
        Return a dict that map each query id to a list of relevant docids
        :param: qrels: Query relevance judgemebt
        :return: A relevance dictionary
        {qid: [doc_ids], ...}
        """
        relevant_docids = {}
        for qid in qrels:
            relevant_docids[qid] = []
            for docid, rel  in qrels[qid].items():                
                if rel > self.relevance_level: 
                    relevant_docids[qid].append(docid)        
        return relevant_docids

    @staticmethod
    def calculate_ranking_metrics(run, qrels):
        """
        Given a run and a qrels, calculate ranking metrics
        :param: run: TREC run of the format {qid: {docid: score}, ...}
        :param: qrels: Query relevance judgement of the format: #TODO: add format
        :return: A dictionary of the format {qid: {measure: value}, ...}
        """
        run_metrics = defaultdict(lambda: defaultdict(lambda: None))
        for metrics in iter_calc([P@1, P@3, P@5, P@10, nDCG@1, nDCG@3, nDCG@5, nDCG@10], qrels, run):
            run_metrics[metrics.query_id][str(metrics.measure)] = metrics.value        
        return run_metrics

    def gather_query_differences(self, run1, run2, qids, qid2diff, metric_name, qid2qrelscores=None):
        """
        Given two runs, and a list of query ids, this function collection and aggreegate information for each query in both run.        
        :param: run1: TREC run of the format {qid: {docid: score}, ...}
        :param: run2: TREC run of the format {qid: {docid: score}, ...}
        :param: qids: A list of query ids (strings) 
        :param: qid2diff: A dict that map query id to a score telling the ranking difference between runs 
        :param: metric_name: Name of the metric used to measure the difference
        :param: qid2qrelscores: A dict formated as {qid: [run1_rel_score, run2_rel_score], ...}
        :return: A list of dicts. Each dict has the following format:
        {
            "fields": {"query_id": "qid", "title": "Title query", "desc": "Can be empty", ... everything else in ir-dataset query},
            "run1": [
                {
                    "doc_id": "id of the doc",
                    "score": <score>,
                    "relevance": <comes from qrels>,
                    "weights": [
                        [field, start, stop, weight]
                        ^ Need more clarity. Return an empty list for now
                    ]

                }
            ],
            "run2": <same format as run 1>
        }
        """              
        relevant_docids = self.get_relevant_docids(self.qrels)

        run1_metrics = self.calculate_ranking_metrics(run1, self.qrels)
        
        if run2:
            run2_metrics = self.calculate_ranking_metrics(run2, self.qrels)

        qids_set = set(qids)  # Sets do O(1) lookups
        qid2info = {}
        
        # each query should have be in a dict of {field: value, ...}, there should be a query_id field.
        for qid in tqdm(qids_set, desc="analyzing queries"):            
            if not qid in self.queries:
                continue 
            query = self.queries[qid]
            # convert to dict if query is not a dict
            if not isinstance(query, dict):                 
                query = query._asdict()                

            # gather top documents in both runs with relevant documents
            run1_doc_ids_to_display = set(list(run1[qid])[:self.num_doc] + relevant_docids[qid])
            if run2:
                run2_doc_ids_to_display =  set(list(run2[qid])[:self.num_doc] + relevant_docids[qid])           
                all_doc_ids_to_display= run1_doc_ids_to_display.union(run2_doc_ids_to_display)
            else:
                all_doc_ids_to_display= run1_doc_ids_to_display
                        
            # add query contrast value
            query["contrast"] = {"name": metric_name, "value": qid2diff[qid]}
            if qid2qrelscores:
                query[f"Run1 {metric_name}"] = qid2qrelscores[qid][0]
                query[f"Run2 {metric_name}"] = qid2qrelscores[qid][1]
            
            # get jugdements for a specific query
            qrels_for_a_query = self.qrels.get(qid, {})

            # gather information to display for top documents
            def collect_display_info_for_each_run(run, doc_ids_to_display, run_idx):                
                if not isinstance(doc_ids_to_display, set):
                    doc_ids_to_display = set(doc_ids_to_display)
                info_to_display = []
                default_field = None
                for rank, (doc_id, score) in enumerate(run[qid].items()):
                    if doc_id not in doc_ids_to_display:
                        continue                     
                    # get the weights for every tokens in the documents
                    doc = self.doc_lookup_func(doc_id)
                    weights = self.weight.score_document_regions(self.queries[qid], doc, run_idx)                                
                    # get snippet to display
                    if default_field == None:
                        default_field = list(filter(lambda x: x != "doc_id", list(doc.keys())))[0]
                    snippet = self.find_snippet(weights, default_field=default_field)

                    info_to_display.append(
                        {
                            "doc_id": doc_id,
                            "score": score,
                            "relevance": qrels_for_a_query.get(doc_id),
                            "rank": rank + 1,
                            "weights": weights,                            
                            "snippet": snippet,
                        }
                    )
                return info_to_display

            run1_to_display = collect_display_info_for_each_run(run1, run1_doc_ids_to_display, 0)
            run2_to_display = []
            if run2:
                run2_to_display = collect_display_info_for_each_run(run2, run2_doc_ids_to_display, 1)

            qid2info[qid] = {
                "fields": query,
                "metrics": {
                    metric: [run1_metrics[qid][metric], run2_metrics[qid][metric]]
                    if run2
                    else [run1_metrics[qid][metric]]
                    for metric in ["P@1", "P@3", "P@5", "P@10", "nDCG@1", "nDCG@3", "nDCG@5", "nDCG@10"]
                },
                "run1": run1_to_display,
                "run2": run2_to_display,
                "doc_ids": list(all_doc_ids_to_display), 
                "summary": self.create_summary(run1_to_display, run2_to_display),
                "mergedWeights": self.merge_weights(run1_to_display, run2_to_display),
            }            
        return qid2info

    def create_summary(self, run1_ranked_docs, run2_ranked_docs):
        """
        Given two run files, create a summary of notable differeces between them,
        :param: run1_ranked_docs: First TREC runs
        :param: run2_ranked_docs: Second TREc runs
        Return a list of strings describing the differences.
        """
        summary = []
        if len(run2_ranked_docs) == 0:
            unjudged_count = 0
            for doc in run1_ranked_docs:
                if doc["relevance"] is None:
                    unjudged_count += 1
            if unjudged_count > 0:
                summary.append(["{} unjudged doc(s) move to a top spot in the ranking".format(unjudged_count)])
            return summary

        run1_doc_map = {doc["doc_id"]: doc for doc in run1_ranked_docs}
        run2_doc_map = {doc["doc_id"]: doc for doc in run2_ranked_docs}
        not_ranked_in_run2 = 0
        for doc_id in run1_doc_map:
            if doc_id not in run2_doc_map:
                not_ranked_in_run2 += 1
        not_ranked_in_run1 = 0
        for doc_id in run2_doc_map:
            if doc_id not in run1_doc_map:
                not_ranked_in_run1 += 1
        not_ranks = []
        if not_ranked_in_run2 > 0:
            not_ranks.append("{} document(s) in run1 not ranked in run2".format(not_ranked_in_run2))
        if not_ranked_in_run1 > 0:
            not_ranks.append("{} document(s) in run2 not ranked in run1".format(not_ranked_in_run1))
        summary.append(not_ranks)
        moves = []
        for doc_id in run1_doc_map:
            if doc_id not in run2_doc_map:
                continue
            rank_in_1 = run1_doc_map[doc_id]["rank"]
            rank_in_2 = run2_doc_map[doc_id]["rank"]
            if rank_in_1 != rank_in_2:
                moves.append(("#{} in run1 moves to #{} in run2".format(rank_in_1, rank_in_2), abs(rank_in_1 - rank_in_2)))
        moves = sorted(moves, key=lambda x: x[1], reverse=True)
        if len(moves) > 0:
            summary.append([move[0] for move in moves[:5]])
        unjudged_run1 = 0
        unjudged_run2 = 0
        for doc_id in run1_doc_map:
            if run1_doc_map[doc_id]["relevance"] is None and run1_doc_map[doc_id]["rank"] <= 10:
                unjudged_run1 += 1
        for doc_id in run2_doc_map:
            if run2_doc_map[doc_id]["relevance"] is None and run2_doc_map[doc_id]["rank"] <= 10:
                unjudged_run2 += 1
        unjudged = []
        if unjudged_run1 > 0:
            unjudged.append("{} unjudged docs in run1 move to a top spot in the ranking".format(unjudged_run1))
        if unjudged_run2 > 0:
            unjudged.append("{} unjudged docs in run2 move to a top spot in the ranking".format(unjudged_run2))
        summary.append(unjudged)
        swap = 0
        doc_ids = list(set(run1_doc_map.keys()).intersection(set(run2_doc_map.keys())))
        for idx1 in range(len(doc_ids)):
            for idx2 in range(idx1 + 1, len(doc_ids)):
                if (run1_doc_map[doc_ids[idx1]]["rank"] - run1_doc_map[doc_ids[idx2]]["rank"]) * (
                    run2_doc_map[doc_ids[idx1]]["rank"] - run2_doc_map[doc_ids[idx2]]["rank"]
                ) < 0:
                    swap += 1
        if swap > 0:
            summary.append(["Number of relative rank reversals: {}".format(swap)])
        return summary

    def merge_weights(self, run1_for_query, run2_for_query):
        """
        Merge the the weights of documents from two run files 
        """
        doc_id2weights = defaultdict(lambda: {"run1": defaultdict(lambda: []), "run2": defaultdict(lambda: [])})

        # If the second run is empty, don't bother to merge
        if not run2_for_query:
            merged_weights = defaultdict(lambda: {})

            for doc in run1_for_query:
                doc_id = doc["doc_id"]
                fields = set(doc_id2weights[doc_id]["run2"].keys()).union(doc_id2weights[doc_id]["run1"])
                fields = set(fields).difference(set(["doc_id"]))

                for field in fields:
                    merged_weights[doc_id][field] = doc["weights"]

            return merged_weights

        for doc in run1_for_query:
            doc_id2weights[doc["doc_id"]]["run1"] = doc["weights"]
        for doc in run2_for_query:
            doc_id2weights[doc["doc_id"]]["run2"] = doc["weights"]

        merged_weights = defaultdict(lambda: {})
        for doc_id in doc_id2weights:
            fields = set(doc_id2weights[doc_id]["run2"].keys()).union(doc_id2weights[doc_id]["run1"])
            fields = set(fields).difference(set(["doc_id"]))           
            for field in fields:
                t = IntervalTree()
                for segment in doc_id2weights[doc_id]["run1"].get(field, []):
                    t.add(Interval(segment[0], segment[1], {"run1": segment[2]}))
                for segment in doc_id2weights[doc_id]["run2"].get(field, []):
                    t.add(Interval(segment[0], segment[1], {"run2": segment[2]}))
                t.split_overlaps()
                t.merge_equals(lambda old_dict, new_dict: old_dict.update(new_dict) or old_dict, {"run1": None, "run2": None})
                merged_intervals = sorted([(i.begin, i.end, i.data) for i in t], key=lambda x: (x[0], x[1]))
                merged_weights[doc_id][field] = merged_intervals            
        return merged_weights

    def find_snippet(self, weights, default_field=None):
        """
        :param weights: A dict of the form {<field_1>: [(start , end, weight), (start, end, weight), ....], <field_2>: ...}
        Fields are document fields from ir_datasets, for eg: 'text'. 'start' and 'end' are character offsets into the doc
        :param default_field: default doc field to display if no matches found.
        :return: A dict {'field': <field_name>, 'start': <start>, 'stop': <end>,  'weights': <weights>}
        """
        MAX_SNIPPET_LEN = 200
        top_field = default_field
        top_snippet_score = -np.inf
        top_range = (-1, -1)
        for field, field_weights in weights.items():
            segments = sorted(field_weights)
            from collections import deque
            seg_queues = deque()
            queue_weights = 0
            sidx, eidx = 0, 0
            for seg in segments:
                eidx += 1
                seg_queues.append(seg)
                queue_weights = queue_weights + seg[2]
                while len(seg_queues) > 1 and (seg_queues[-1][0] - MAX_SNIPPET_LEN > seg_queues[0][1]):
                    queue_weights = queue_weights - seg_queues[0][2]
                    seg_queues.popleft()
                    sidx += 1
                if queue_weights > top_snippet_score:
                    # _logger.info(top_snippet_score)
                    top_snippet_score = queue_weights
                    top_range = (sidx, eidx)
                    top_field = field
        # reconstruct the snippet
        if top_snippet_score > 0:
            snp_weights = sorted(weights[top_field])[top_range[0] : top_range[1]]
            # start = max(snp_weights[0][1] - max(0, (MAX_SNIPPET_LEN - snp_weights[-1][0] + snp_weights[0][1])/2), 0)
            start = max(snp_weights[0][0] - 5, 0)
            stop = start + MAX_SNIPPET_LEN
            snp_weights = [(w[0] - start, w[1] - start, w[2]) for w in snp_weights if w[0] < stop]
            snp_weights = sorted(snp_weights)
            top_snippet = {"field": top_field, "start": start, "stop": stop, "weights": snp_weights}
        else:
            #fall back onto the default field
            top_snippet = {"field": default_field, "start": 0, "stop": MAX_SNIPPET_LEN, "weights": []}
        return top_snippet

    def retrieve_document_to_display(self, doc_ids_to_display):
        """        
        From the given list of document ids, retrieve the doc dictionary storing content
        :param doc_ids_to_display: list of document ids to display        
        :return: A dict of the form:
        {
            <doc_id>: {"doc_id": <doc_id>, "text": <content of the doc> ... }
            ...
        }
        """
        docs_to_return = {}
        for doc_id in _logger.pbar(
            doc_ids_to_display, desc="Docs iter", total=len(doc_ids_to_display)
        ):
            doc = self.doc_lookup_func(doc_id)
            if not isinstance(doc, dict):
                #TODO: convert doc object to dictionary
                doc = doc._asdict()
            docs_to_return[doc_id] = doc
        return docs_to_return

    def json(self, run1_fn, run2_fn=None):
        """
        Represent the data to be visualized in a json format.
        The format is specified here: https://github.com/capreolus-ir/diffir-private/issues/5
        :params: 2  TREC runs. These dicts of the form {qid: {docid: score}}
        """
        run1 = load_trec_run(run1_fn)
        run2 = load_trec_run(run2_fn) if run2_fn is not None else None        
        
        diff_queries, qid2diff, metric_name, qid2qrelscores = self.measure.query_differences(run1, run2, qrels=self.qrels)
        
        qid2querydiff = self.gather_query_differences(
            run1, run2, diff_queries, qid2diff, metric_name, qid2qrelscores=qid2qrelscores
        )

        doc_ids_to_fetch = set()
        for qid in qid2querydiff:        
            doc_ids_to_fetch.update(qid2querydiff[qid]["doc_ids"])

        doc_objects = self.retrieve_document_to_display(doc_ids_to_fetch)

        return json.dumps(
            {
                "meta": {
                    "run1_name": run1_fn,
                    "run2_name": run2_fn,
                    "dataset": self.dataset,
                    "measure": self.measure.module_name,
                    # "weight": self.weight.module_name,
                    "qrelDefs": self.qrel_defs,
                    "queryFields": self.qfields,
                    "qfield_to_use": self.qfield_to_use,
                    # "docFields": dataset.docs_cls()._fields,
                    "relevanceColors": self.make_rel_colors(),
                },
                "queries": [qid2querydiff[qid] for qid in qid2querydiff],
                "docs": doc_objects,
            }
        )

    def print_query_to_console(self, q, console):
        """
        Print query fields to the console
        """
        # print query to console using rich
        fields = [f"# [blue][bold]{k}: [white][italic]{v}".replace("\n", " ") for k, v in q["fields"].items()]
        query_text = "\n".join(fields)
        query_panel = Panel(query_text, title="Query # {}".format(q["fields"]["query_id"]), expand=True)
        console.print(query_panel)

    def render_snippet_for_cli(self, doc_id: str, snp, docs):
        """
        Print snippet of a document to the console 
        :param: doc_id: document id
        :param: snp: snippet contaiting the document fields, and the weights
        :param: docs: a dictionary that map document ids to contents
        """
        snp_text = docs[doc_id][snp["field"]][snp["start"] : snp["stop"]]
        # Escape special characters 
        snp_text = snp_text.replace("[", "")
        idx_change = 0
        for s, e, w in snp["weights"]:
            s = s + idx_change
            e = e + idx_change
            snp_text = snp_text[:s] + f"[underline]{snp_text[s:e]}[/underline]" + snp_text[e:]
            idx_change += 23

        return snp_text

    def cli_display_one_run(self, console, q, start_idx, end_idx, docs, run1_name):
        """
        Visualize a single run
        :param: console: console object to display
        :param: q: query to display
        :param: start_idx: start document index to display
        :param: end_idx: end document index to display
        :param: docs: document dictioary maps document id to contents
        :param: run1_name: name of the the TREC run
        """
        docid2rank_run1 = defaultdict(lambda: "Not ranked")
        for doc in q["run1"]:
            docid2rank_run1[doc["doc_id"]] = doc["rank"]

        table = Table(show_header=True, header_style="bold red", title=run1_name, show_lines=True, expand=True)
        table.add_column("DocID", justify="center", style="cyan", no_wrap=True)
        table.add_column("Ranking", justify="left", style="magenta")
        table.add_column("Rel", justify="center", style="green")
        table.add_column("Snippet", justify="full", style="white")

        def handle_non(rel):
            retval = "Unjudged" if rel is None else str(rel)
            return retval

        for run1_doc in q["run1"][start_idx:end_idx]:
            snippet = self.render_snippet_for_cli(run1_doc["doc_id"], run1_doc["snippet"], docs)
            table.add_row(
                run1_doc["doc_id"],
                "[bold]Score[/bold]: {}\n[bold]Rank[/bold]:{}\b".format(run1_doc["score"], str(run1_doc["rank"])),
                handle_non(run1_doc["relevance"]),
                snippet,
            )

        self.print_query_to_console(q, console)
        console.print(table)

    def cli_compare_two_runs(self, console, q, start_idx, end_idx, docs, run1_name, run2_name):
        """
        Visualize two runs with the command line interface
        :param: console: The target console to print
        :param: q: query to display
        :param: start_idx: start document index 
        :param: end_idx: end document index
        :param: docs: a dictionary that maps document id to document contents
        :param: run1_name: first TREC run
        ;param: run2_name: second TREC run 
        """
        docid2rank_run1 = defaultdict(lambda: "Not ranked")
        for doc in q["run1"]:
            docid2rank_run1[doc["doc_id"]] = doc["rank"]

        docid2rank_run2 = defaultdict(lambda: "Not ranked")
        for doc in q["run2"]:
            docid2rank_run2[doc["doc_id"]] = doc["rank"]

        self.print_query_to_console(q, console)

        # rprint(query_panel)
        table = Table(
            show_header=True,
            header_style="bold red",
            title="Comparision [bold]run#1:[/bold] [red]{}[/red] vs [bold]run#2:[/bold] [red]{}[/red]".format(
                run1_name, run2_name
            ),
            show_lines=True,
            expand=True,
        )
        table.add_column("DocID", justify="center", style="cyan", no_wrap=True)
        table.add_column("Run #1", justify="left", style="magenta")
        table.add_column("Rel", justify="center", style="green")
        table.add_column("Snippet", justify="full", style="white")
        table.add_column(" ")
        table.add_column("DocID", justify="left", style="cyan", no_wrap=True)
        table.add_column("Run #2", justify="left", style="magenta")
        table.add_column("Rel", justify="center", style="green")
        table.add_column("Snippet", justify="full", style="white")

        def handle_non(rel):
            if rel == None:
                return "Unjudged"
            else:
                return str(rel)

        for run1_doc, run2_doc in zip(q["run1"][start_idx:end_idx], q["run2"][start_idx:end_idx]):
            snp_1 = self.render_snippet_for_cli(run1_doc["doc_id"], run1_doc["snippet"], docs)
            snp_2 = self.render_snippet_for_cli(run2_doc["doc_id"], run2_doc["snippet"], docs)
            table.add_row(
                run1_doc["doc_id"],
                "[bold]Score[/bold]: {}\n[bold]Rank[/bold]:{}\n[bold]Rank in run#2[/bold]: {}".format(
                    run1_doc["score"], str(run1_doc["rank"]), docid2rank_run2[run1_doc["doc_id"]]
                ),
                handle_non(run1_doc["relevance"]),
                snp_1,
                " ",
                run2_doc["doc_id"],
                "[bold]Score[/bold]: {}\n[bold]Rank[/bold]:{}\n[bold]Rank in run#1[/bold]: {}".format(
                    run2_doc["score"], str(run2_doc["rank"]), docid2rank_run1[run2_doc["doc_id"]]
                ),
                handle_non(run2_doc["relevance"]),
                snp_2,
            )
        console.print(table)

    def cli(self, runs):
        """
        Command line interface
        """
        json_data = json.loads(self.json(*runs))
        # 1. render the json_data into the terminal
        query_text = ""
        queries = json_data["queries"]
        docs = json_data["docs"]
        current_index = 0

        console = Console()
        if len(runs) == 2:
            for current_index in range(len(queries)):
                self.cli_compare_two_runs(
                    console, queries[current_index], 0, None, docs, json_data["meta"]["run1_name"], json_data["meta"]["run2_name"]
                )
                ans = Confirm.ask("Want to see the next query?")
                if not ans:
                    return
        else:
            for current_index in range(len(queries)):
                self.cli_display_one_run(console, queries[current_index], 0, None, docs, json_data["meta"]["run1_name"])

    def web(self, runs):
        """
        Web interface
        """
        json_data = self.json(*runs)

        script_dir = os.path.dirname(__file__)
        template = Template(filename=os.path.join(script_dir, "templates", "template.html"))

        return template.render(data=json_data)

    def make_rel_colors(self):
        result = {None: "#888888"}
        if not self.qrel_defs:
            return result
        NON_POS_COLORS = {
            0: [],
            1: ["#d54541"],  # red
            2: ["#6c272a", "#d54541"],  # dark red, red
            3: ["#6c272a", "#d54541", "#c7797a"],  # dark red, red, light red
        }
        POS_COLORS = {
            0: [],
            1: ["#52b262"],  # green
            2: ["#52b262", "#58aadc"],  # green, blue
            3: ["#caab39", "#52b262", "#58aadc"],  # yellow, green, blue
            4: ["#cf752b", "#caab39", "#52b262", "#58aadc"],  # orange, yellow, green, blue
            5: ["#cf752b", "#caab39", "#52b262", "#58aadc", "#8a5fd4"],  # orange, yellow, green, blue, purple
        }        
        nonpos = sorted([k for k in self.qrel_defs.keys() if k <= 0])
        if len(nonpos) in NON_POS_COLORS:
            result.update(zip(nonpos, NON_POS_COLORS[len(nonpos)]))
        else:
            result.update(zip(nonpos, NON_POS_COLORS[3][0] * (len(nonpos) - 3) + NON_POS_COLORS[3]))
        pos = sorted([k for k in self.qrel_defs.keys() if k > 0])
        if len(pos) in POS_COLORS:
            result.update(zip(pos, POS_COLORS[len(pos)]))
        else:
            result.update(zip(pos, POS_COLORS[3] + (POS_COLORS[5][-1] * (len(nonpos) - 5))))
        return result


if __name__ == "__main__":
    main()

# diffir runa runb --dataset [irds_id] --queries [optional query file (TSV?), or uses irds] --measure [recip_rank, some rank correlation method, etc.] --docweight [the above term/passage/doc weight format?]
