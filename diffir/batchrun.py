import argparse
import os
import sys
from pathlib import Path

import ir_datasets

from diffir.run import diff
from diffir.utils import load_trec_run

_logger = ir_datasets.log.easy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory")
    parser.add_argument("-o", "--output", dest="output_dir")
    # parser.add_argument("-c", "--cli", dest="cli", action="store_true")
    # parser.add_argument("-w", "--web", dest="web", action="store_true")
    parser.add_argument("--config", dest="config", nargs="*")

    args = parser.parse_args()
    indir = Path(args.directory)
    output = Path(args.output_dir) if args.output_dir else indir / "diffir"
    output.mkdir(exist_ok=True)

    single_runs = []
    weight_files = {}
    for fn in indir.iterdir():
        if fn.suffix == ".diffir":
            _logger.debug(f"skipping weights file: {fn}")
            continue
        try:
            # TODO to check whether the file is valid, we call load_trec_run both here and in diffir.run
            load_trec_run(fn)
            potential_weights_file = "{}.diffir".format(os.path.basename(fn).split(".")[0])
            if os.path.isfile(os.path.join(indir, potential_weights_file)):
                weight_files[fn.as_posix()] = os.path.join(indir, potential_weights_file)

            single_runs.append(fn.as_posix())
        except:
            _logger.warn(f"failed to parse run: {fn}")

    processed_pairs = set()
    for fn1 in sorted(single_runs):
        for fn2 in single_runs:
            if fn1 == fn2:
                continue
            if fn1 > fn2:
                fn1, fn2 = fn2, fn1

            if (fn1, fn2) in processed_pairs:
                continue

            config = args.config
            if fn1 in weight_files:
                config += ["weight.name=custom", "weight.weights_1={}".format(weight_files[fn1])]
            if fn2 in weight_files:
                config += ["weight.name=custom", "weight.weights_2={}".format(weight_files[fn2])]

            task_config, html = diff([fn1, fn2], config, cli=False, web=True, print_html=False)
            output_fn = output / task_config["dataset"]
            os.makedirs(output_fn, exist_ok=True)

            with open(output_fn / (os.path.basename(fn1) + "___" + os.path.basename(fn2) + ".html"), "wt") as outf:
                print(html, file=outf)

            processed_pairs.add((fn1, fn2))

    for fn1 in single_runs:
        config = args.config
        if fn1 in weight_files:
            config += ["weight.name=custom", "weight.weights_1={}".format(weight_files[fn1])]

        task_config, html = diff([fn1], config, cli=False, web=True, print_html=False)
        output_fn = output / task_config["dataset"]
        os.makedirs(output_fn, exist_ok=True)

        with open(output_fn / (os.path.basename(fn1) + ".html"), "wt") as outf:
            print(html, file=outf)

        with open(output_fn / "runs.txt", "wt") as outf:
            for run in sorted(single_runs):
                print(run.split("/")[-1], file=outf)


if __name__ == "__main__":
    main()
