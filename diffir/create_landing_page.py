import argparse
from markupsafe import Markup
import json
from collections import defaultdict
import os
from jinja2 import Environment, PackageLoader, select_autoescape


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inputdir")

    args = parser.parse_args()

    datasets = []
    runfiles = defaultdict(list)
    for dirname in os.listdir(args.inputdir):
        if not os.path.isdir(os.path.join(args.inputdir, dirname)):
            continue

        dataset_name = dirname
        datasets.append(dataset_name)

        with open(os.path.join(args.inputdir, dataset_name, "runs.txt")) as f:
            for filename in f:
                print(filename.strip())
                runfiles[dataset_name].append(filename.strip())

    env = Environment(loader=PackageLoader("diffir", "templates"), autoescape=select_autoescape(["html", "xml"]))

    landing_template = env.get_template("landing.html")
    with open(os.path.join(args.inputdir, "index.html"), "wt") as outf:
        print(landing_template.render(datasets=datasets, runfiles=runfiles, rawdata=Markup(json.dumps(runfiles))), file=outf)
