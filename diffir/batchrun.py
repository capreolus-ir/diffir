import argparse
import itertools
import multiprocessing
import os
from functools import partial
from pathlib import Path

import ir_datasets

from diffir.run import diff
from diffir.utils import load_trec_run

_logger = ir_datasets.log.easy()


def process_runs(fns, config, output):
    task_config, html = diff(fns, config, cli=False, web=True, print_html=False)
    outdir = output / task_config["dataset"]
    outdir.mkdir(exist_ok=True, parents=True)

    outfn = outdir / ("___".join(os.path.basename(x) for x in fns) + ".html")
    with open(outfn, "wt") as outf:
        print(html, file=outf)

    return outdir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("-o", "--output", dest="output_dir")
    parser.add_argument("--config", dest="config", nargs="*")

    args = parser.parse_args()
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
    f = partial(process_runs, config=args.config, output=output)
    with multiprocessing.Pool(8) as p:
        outdirs = p.map(f, queue)

    with open(outdirs[0] / "runs.txt", "wt") as outf:
        for run in sorted(single_runs):
            print(run.split("/")[-1], file=outf)


if __name__ == "__main__":
    main()
