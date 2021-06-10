import argparse
import itertools
import multiprocessing
import os
from functools import partial
from pathlib import Path
from markupsafe import Markup
import json
from collections import defaultdict
from jinja2 import Environment, PackageLoader, select_autoescape

import ir_datasets

from diffir.run import diff
from diffir.utils import load_trec_run

_logger = ir_datasets.log.easy()


def process_runs(fns, config, output):
    task_config, html = diff(fns, config, cli=False, web=True, print_html=False)

    dataset_name = task_config["dataset"].replace("/", ".")
    measure_name = task_config["measure"] if task_config["measure"] != "qrel" else "qrel." + task_config["metric"]
    outdir = output / f"{dataset_name}---measure___{measure_name}"
    outdir.mkdir(exist_ok=True, parents=True)

    outfn = outdir / ("___".join(os.path.basename(x) for x in fns) + ".html")
    with open(outfn, "wt") as outf:
        print(html, file=outf)

    return outdir


def regenerate_landing_page(outdir):
    datasets = []
    name_dict = {}
    runfiles = defaultdict(list)
    for dirname in os.listdir(outdir):
        if not os.path.isdir(os.path.join(outdir, dirname)):
            continue

        dataset_name = dirname
        configs = dataset_name.split("---")
        display_name = configs[0] + (
            "" if len(configs) == 1 else " (" + ",".join([c.replace("___", ":") for c in configs[1:]]) + ")"
        )
        name_dict[dataset_name] = display_name
        datasets.append(dataset_name)

        with open(os.path.join(outdir, dataset_name, "runs.txt")) as f:
            for filename in f:
                print(filename.strip())
                runfiles[dataset_name].append(filename.strip())

    env = Environment(loader=PackageLoader("diffir", "templates"), autoescape=select_autoescape(["html", "xml"]))

    landing_template = env.get_template("landing.html")
    with open(os.path.join(outdir, "index.html"), "wt") as outf:
        print(
            landing_template.render(
                datasets=datasets, name_dict=name_dict, runfiles=runfiles, rawdata=Markup(json.dumps(runfiles))
            ),
            file=outf,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("-o", "--output", dest="output_dir")
    parser.add_argument("--dataset", dest="dataset", type=str, help="dataset from ir_datasets")
    parser.add_argument(
        "--measure", dest="measure", type=str, default="tauap", help="measure for ranking difference (qrel, tauap,weightedtau)"
    )
    parser.add_argument("--metric", dest="metric", type=str, default="MAP", help="metric used with qrel measure")
    parser.add_argument("--topk", dest="topk", type=int, default=10)
    args = parser.parse_args()
    config = {
        "dataset": args.dataset,
        "measure": args.measure,
        "metric": args.metric,
        "topk": args.topk,
        "weight": {"weights_1": None, "weights_2": None},
    }
    indir = Path(args.directory)
    output = Path(args.output_dir) if args.output_dir else indir / "diffir"
    output.mkdir(exist_ok=True)

    single_runs = []
    for fn in indir.iterdir():
        if fn.suffix == ".diffir":
            _logger.debug(f"skipping weights file: {fn}")
            continue
        try:
            # TODO to check whether the file is valid, we call load_trec_run both here and in diffir.run
            load_trec_run(fn)
            single_runs.append(fn.as_posix())
        except:
            _logger.warn(f"failed to parse run: {fn}")

    single_runs = sorted(single_runs)  # sorted needed for itertools ordering
    queue = [(fn,) for fn in single_runs] + list(itertools.combinations(single_runs, 2))
    f = partial(process_runs, config=config, output=output)
    with multiprocessing.Pool(8) as p:
        outdirs = p.map(f, queue)

    with open(outdirs[0] / "runs.txt", "wt") as outf:
        for run in sorted(single_runs):
            print(run.split("/")[-1], file=outf)

    regenerate_landing_page(output)


if __name__ == "__main__":
    main()
