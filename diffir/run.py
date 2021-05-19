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

from diffir import QrelMeasure, TopkMeasure, CustomWeight, ExactMatchWeight
from diffir.utils import load_trec_run

_logger = ir_datasets.log.easy()


def main():
    help = """
            Usage:
              run.py RUN1 RUN2 [options] [([with] CONFIG...)]
              run.py (-h | --help)


            Options:
              -c --cli                      CLI mode (default)
              -w --web                      webui mode
              -h --help                     Print this help message and exit.


            Arguments:
              RUN1      First run file to compare
              RUN2      Second run file to compare
              CONFIG    Configuration assignments of the form foo.bar=17

           """

    parser = argparse.ArgumentParser()
    parser.add_argument("runfiles", nargs="+")
    parser.add_argument("-c", "--cli", dest="cli", action="store_true")
    parser.add_argument("-w", "--web", dest="web", action="store_true")
    parser.add_argument("--dataset", dest="dataset", type=str)
    parser.add_argument("--measure", dest="measure", type=str, default="qrel")
    parser.add_argument("--mmetric", dest="mmetric", type=str, default="map")
    parser.add_argument("--mtopk", dest="mtopk", type=int, default="3")
    parser.add_argument("--weights_1", dest="weights_1", type=str, default=None, required=False)
    parser.add_argument("--weights_2", dest="weights_2", type=str, default=None, required=False)
    # parser.add_argument("--config", dest="config", nargs="*")
    args = parser.parse_args()
    config = {"dataset": args.dataset, "measure": args.measure, "mmetric": args.mmetric, "mtopk": args.mtopk,
                "weight": {"weights_1": args.weights_1, "weights_2": args.weights_2}}
    diff(args.runfiles, config, cli=args.cli, web=args.web,)


def diff(runs, config, cli, web, print_html=True):
    for i, run in enumerate(runs):
        if config["weight"][f"weights_{i + 1}"] is None:
            if os.path.exists(run + ".diffir"):
                config["weight"][f"weights_{i + 1}"] = run + ".diffir"
    task = MainTask(**config)
    if cli:
        task.cli(runs)
    if web:
        html = task.web(runs)
        if print_html:
            print(html)
        return config, html


class MainTask:
    module_type = "task"
    module_name = "main"

    def __init__(self, dataset="none", queries="none", measure="topk", mmetric="weighted_tau", mtopk=3, weight={}):
        self.dataset = dataset
        self.queries = queries
        if measure == "qrels":
            self.measure = QrelMeasure(mmetric, mtopk)
        else:
            self.measure = TopkMeasure(mmetric, mtopk)
        if weight["weights_1"] or weight["weights_2"]:
            self.weight = CustomWeight(weight["weights_1"], weight["weights_2"])
        else:
            self.weight = ExactMatchWeight()

    def create_query_objects(self, run_1, run_2, qids, qid2diff, metric_name, dataset):
        """
        TODO: Need a better name
        This method takes in 2 runs and a set of qids, and constructs a dict for each qid (format specified below)
        :param: run_1: TREC run of the format {qid: {docid: score}, ...}
        :param: run_2: TREC run of the format {qid: {docid: score}, ...}
        :param qids: A list of qids (strings)
        :param dataset: Instance of an ir-datasets object
        :return: A list of dicts. Each dict has the following format:
        {
            "fields": {"query_id": "qid", "title": "Title query", "desc": "Can be empty", ... everything else in ir-dataset query},
            "run_1": [
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
            "run_2": <same format as run 1>
        }
        """
        assert dataset.has_qrels(), "Cannot determine whether the doc is relevant - need qrels"
        qrels = dataset.qrels_dict()

        docstore = dataset.docs_store()

        qids_set = set(qids)  # Sets do O(1) lookups
        qid2object = {}
        for query in tqdm(dataset.queries_iter(), desc="analyzing queries"):
            if query.query_id not in qids_set:
                continue

            RESULT_COUNT = 10
            doc_ids = (
                set(list(run_1[query.query_id])[:RESULT_COUNT] + list(run_2[query.query_id])[:RESULT_COUNT])
                if run_2
                else list(run_1[query.query_id])[:RESULT_COUNT]
            )

            fields = query._asdict()
            fields["metric"] = {"name": metric_name, "value": qid2diff[query.query_id]}
            qrels_for_query = qrels.get(query.query_id, {})
            run_1_for_query = []
            for rank, (doc_id, score) in enumerate(run_1[query.query_id].items()):
                if doc_id not in doc_ids:
                    continue
                doc = docstore.get(doc_id)
                weights = self.weight.score_document_regions(query, doc, 0)
                run_1_for_query.append(
                    {
                        "doc_id": doc_id,
                        "score": score,
                        "relevance": qrels_for_query.get(doc_id),
                        "rank": rank + 1,
                        "weights": weights,
                        "snippet": self.find_snippet(weights, doc),
                    }
                )

            run_2_for_query = []

            if run_2 is not None:
                for rank, (doc_id, score) in enumerate(run_2[query.query_id].items()):
                    if doc_id not in doc_ids:
                        continue
                    doc = docstore.get(doc_id)
                    weights = self.weight.score_document_regions(query, doc, 1)
                    run_2_for_query.append(
                        {
                            "doc_id": doc_id,
                            "score": score,
                            "relevance": qrels_for_query.get(doc_id),
                            "rank": rank + 1,
                            "weights": weights,
                            "snippet": self.find_snippet(weights, doc),
                        }
                    )

            qid2object[query.query_id] = {
                "fields": fields,
                "run_1": run_1_for_query,
                "run_2": run_2_for_query,
                "mergedWeights": self.merge_weights(run_1_for_query, run_2_for_query),
            }

        return [qid2object[id] for id in qids]

    def merge_weights(self, run1_for_query, run_2_for_query):
        doc_id2weights = defaultdict(lambda: {"run1": defaultdict(lambda: []), "run2": defaultdict(lambda: [])})

        # If the second run is empty, don't bother to merge
        if not run_2_for_query:
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
        for doc in run_2_for_query:
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
                t.merge_equals(lambda old_dict, new_dict: old_dict.update(new_dict) or old_dict,
                               {"run1": None, "run2": None})
                merged_intervals = sorted([(i.begin, i.end, i.data) for i in t], key=lambda x: (x[0], x[1]))
                merged_weights[doc_id][field] = merged_intervals

        return merged_weights

    def find_snippet(self, weights, doc):
        """
        :param weights: A dict of the form {<field_1>: [(start , end, weight), (start, end, weight), ....], <field_2>: ...}
        Fields are document fields from ir_datasets, for eg: 'text'. 'start' and 'end' are character offsets into the doc
        :param doc: A large string representing the doc
        :return: A dict {'field': <field_name>, 'start': <start>, 'stop': <end>,  'weights': <weights>}
        """
        MAX_SNIPPET_LEN = 200
        top_field = None
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
            snp_weights = sorted(weights[top_field])[top_range[0]: top_range[1]]
            # start = max(snp_weights[0][1] - max(0, (MAX_SNIPPET_LEN - snp_weights[-1][0] + snp_weights[0][1])/2), 0)
            start = max(snp_weights[0][0] - 5, 0)
            stop = start + MAX_SNIPPET_LEN
            snp_weights = [(w[0] - start, w[1] - start, w[2]) for w in snp_weights if w[0] < stop]
            snp_weights = sorted(snp_weights)
            top_snippet = {"field": top_field, "start": start, "stop": stop, "weights": snp_weights}
        else:
            # fall back onto the first text
            top_snippet = {"field": doc._fields[1], "start": 0, "stop": MAX_SNIPPET_LEN, "weights": []}
        return top_snippet

    def create_doc_objects(self, query_objects, dataset):
        """
        TODO: Need a better name
        From the given query objects, fetch the used docids from the ir-dataset object
        :param query_objects: The return type of `create_query_objects()`
        :param dataset: An instance of irdatasets
        :return: A dict of the form:
        {
            <doc_id>: {"doc_id": <doc_id>, "text": <content of the doc>, "url": <url>, ... rest from ir-dataset,
            ...
        }
        """
        doc_objects = {}

        doc_ids_to_fetch = set()
        for query_obj in query_objects:
            run_1 = query_obj["run_1"]
            run_2 = query_obj["run_2"]

            for listed_doc in run_1 + run_2:
                doc_ids_to_fetch.add(listed_doc["doc_id"])

        for doc in _logger.pbar(
                dataset.docs_store().get_many_iter(doc_ids_to_fetch), desc="Docs iter", total=len(doc_ids_to_fetch)
        ):
            doc_objects[doc.doc_id] = doc._asdict()

        return doc_objects

    def json(self, run_1_fn, run_2_fn=None):
        """
        Represent the data to be visualized in a json format.
        The format is specified here: https://github.com/capreolus-ir/diffir-private/issues/5
        :params: 2  TREC runs. These dicts of the form {qid: {docid: score}}
        """
        run_1 = load_trec_run(run_1_fn)
        run_2 = load_trec_run(run_2_fn) if run_2_fn is not None else None

        dataset = ir_datasets.load(self.dataset)
        assert dataset.has_docs()
        assert dataset.has_queries()
        # TODO: handle the case without qrels
        assert dataset.has_queries()

        diff_queries, qid2diff, metric_name = self.measure.query_differences(run_1, run_2, dataset=dataset)
        # _logger.info(diff_queries)
        diff_query_objects = self.create_query_objects(run_1, run_2, diff_queries, qid2diff, metric_name, dataset)
        doc_objects = self.create_doc_objects(diff_query_objects, dataset)

        return json.dumps(
            {
                "meta": {
                    "run1_name": run_1_fn,
                    "run2_name": run_2_fn,
                    "dataset": self.dataset,
                    "measure": self.measure.module_name,
                    "weight": self.weight.module_name,
                    "qrelDefs": dataset.qrels_defs(),
                    "queryFields": dataset.queries_cls()._fields,
                    "docFields": dataset.docs_cls()._fields,
                    "relevanceColors": self.make_rel_colors(dataset),
                },
                "queries": diff_query_objects,
                "docs": doc_objects,
            }
        )

    def print_query_to_console(self, q, console):
        # print query to console using rich
        fields = [f"# [blue][bold]{k}: [white][italic]{v}".replace("\n", " ") for k, v in q["fields"].items()]
        query_text = "\n".join(fields)
        query_panel = Panel(query_text, title="Query # {}".format(q["fields"]["query_id"]), expand=True)
        console.print(query_panel)

    def render_snippet_for_cli(self, doc_id, snp, docs):
        snp_text = docs[doc_id][snp["field"]][snp["start"]: snp["stop"]]
        idx_change = 0
        for s, e, w in snp["weights"]:
            s = s + idx_change
            e = e + idx_change
            snp_text = snp_text[:s] + f"[underline]{snp_text[s:e]}[/underline]" + snp_text[e:]
            idx_change += 23

        return snp_text

    def cli_display_one_query(self, console, q, start_idx, end_idx, docs, run_1_name):
        docid2rank_run1 = defaultdict(lambda: "Not ranked")
        for doc in q["run_1"]:
            docid2rank_run1[doc["doc_id"]] = doc["rank"]

        table = Table(show_header=True, header_style="bold red", title=run_1_name, show_lines=True, expand=True)
        table.add_column("DocID", justify="center", style="cyan", no_wrap=True)
        table.add_column("Ranking", justify="left", style="magenta")
        table.add_column("Rel", justify="center", style="green")
        table.add_column("Snippet", justify="full", style="white")

        def handle_non(rel):
            retval = "Unjudged" if rel is None else str(rel)
            return retval

        for run1_doc in q["run_1"][start_idx:end_idx]:
            snippet = self.render_snippet_for_cli(run1_doc["doc_id"], run1_doc["snippet"], docs)
            table.add_row(
                run1_doc["doc_id"],
                "[bold]Score[/bold]: {}\n[bold]Rank[/bold]:{}\b".format(run1_doc["score"], str(run1_doc["rank"])),
                handle_non(run1_doc["relevance"]),
                snippet,
            )

        self.print_query_to_console(q, console)
        console.print(table)

    def cli_compare_one_query(self, console, q, start_idx, end_idx, docs, run1_name, run2_name):
        docid2rank_run1 = defaultdict(lambda: "Not ranked")
        for doc in q["run_1"]:
            docid2rank_run1[doc["doc_id"]] = doc["rank"]

        docid2rank_run2 = defaultdict(lambda: "Not ranked")
        for doc in q["run_2"]:
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

        for run1_doc, run2_doc in zip(q["run_1"][start_idx:end_idx], q["run_2"][start_idx:end_idx]):
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
        json_data = json.loads(self.json(*runs))
        # 1. render the json_data into the terminal
        query_text = ""
        queries = json_data["queries"]
        docs = json_data["docs"]
        current_index = 0

        console = Console()
        if len(runs) == 2:
            for current_index in range(len(queries)):
                self.cli_compare_one_query(
                    console, queries[current_index], 0, None, docs, json_data["meta"]["run1_name"],
                    json_data["meta"]["run2_name"]
                )
                ans = Confirm.ask("Want to see the next query?")
                if not ans:
                    return
        else:
            with console.pager():
                for current_index in range(len(queries)):
                    self.cli_display_one_query(console, queries[current_index], 0, None, docs,
                                               json_data["meta"]["run1_name"])

    def web(self, runs):
        json_data = self.json(*runs)

        script_dir = os.path.dirname(__file__)
        template = Template(filename=os.path.join(script_dir, "templates", "template.html"))

        return template.render(data=json_data)

    def make_rel_colors(self, dataset):
        result = {None: "#888888"}
        if not dataset.has_qrels():
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
        qrel_defs = dataset.qrels_defs()
        nonpos = sorted([k for k in qrel_defs.keys() if k <= 0])
        if len(nonpos) in NON_POS_COLORS:
            result.update(zip(nonpos, NON_POS_COLORS[len(nonpos)]))
        else:
            result.update(zip(nonpos, NON_POS_COLORS[3][0] * (len(nonpos) - 3) + NON_POS_COLORS[3]))
        pos = sorted([k for k in qrel_defs.keys() if k > 0])
        if len(pos) in POS_COLORS:
            result.update(zip(pos, POS_COLORS[len(pos)]))
        else:
            result.update(zip(pos, POS_COLORS[3] + (POS_COLORS[5][-1] * (len(nonpos) - 5))))
        return result


if __name__ == "__main__":
    main()

# diffir runa runb --dataset [irds_id] --queries [optional query file (TSV?), or uses irds] --measure [recip_rank, some rank correlation method, etc.] --docweight [the above term/passage/doc weight format?]
