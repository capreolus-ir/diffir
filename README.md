
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation Status](https://readthedocs.org/projects/diffir/badge/?version=latest)](https://capreolus.readthedocs.io/projects/diffir/?badge=latest)
[![Worfklow](https://github.com/capreolus-ir/diffir/workflows/test/badge.svg)](https://github.com/capreolus-ir/diffir/actions)
[![PyPI version fury.io](https://badge.fury.io/py/diffir.svg)](https://pypi.python.org/pypi/diffir/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black) 

# DiffIR
DiffIR is a tool for visually 'diffing' the difference between two sets of rankings. Given a pair of TREC runs containing rankings for multiple queries, DiffIR identifies contrasting queries that have "substantially" different results between the two systems and generates a visual side-by-side comparison illustrating how the key rankings differ.

DiffIR supports multiple *query contrast meastures* for ranking comparison including unsupervised ranking correlations like TauAP and supervised comparison based on existing judgments. DiffIR additionally accepts term importance weights in order to highlight the terms most relevant to a model's relevance prediction.

## Usage [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1MrmY1lKa0Pru--gqAqrcNsilNKZjbm1y/)
### Installation
Python 3 is required. Install via PyPI:

```
pip install diffir
```

### Usage
Download two run files to test with:
```
wget -c https://github.com/capreolus-ir/diffir/raw/master/trec-dl-2020/p_bm25
wget -c https://github.com/capreolus-ir/diffir/raw/master/trec-dl-2020/p_bm25rm3
```

Compare the two files and output a comparison page to `bm25_bm25rm3.html`:
```
diffir p_bm25 p_bm25rm3 -w --dataset msmarco-passage/trec-dl-2020 \
       --measure qrel --metric nDCG@5 --topk 3 > bm25_bm25rm3.html
```
Now open `bm25_bm25rm3.html` in your web browser. You should see DiffIR's web interface:

<a href="https://raw.githubusercontent.com/capreolus-ir/diffir/master/docs/images/screenshot.png"><img src="https://raw.githubusercontent.com/capreolus-ir/diffir/master/docs/images/screenshot.png"></a>

### Command line arguments
Usage: `diffir <run files> <options>` 
where the run files are 1 or 2 positional arguments indicating the run files to visualize, and `<options>` are:
- `-w` to output HTML or `-c` for the command line interface
- `--dataset <id>`: a dataset id from [ir_datasets](https://ir-datasets.com/)
- `--measure <measure>` the query contrast measure to use. Valid measures: qrel, [tauap](https://dl.acm.org/doi/10.1145/1390334.1390435), [pearsonrank](https://dl.acm.org/doi/10.1145/2911451.2914728), [weightedtau](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.weightedtau.html), [spearmanr](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.spearmanr.html), kldiv (using scores)
- `--metric <metric>`: the relevance metric to use with the qrel measure. Accepts [ir_measures](https://github.com/terrierteam/ir_measures) notation
- `--topk <k>`: the number of queries to compare (as identified by the query contrast measure)
- `--weights_1 <file>`, `--weights_2 <file>`: term importance files to use for snippet selection

### Batch mode
Use `diffir-batch` to generate comparison pages for every pair of run files in a directory.

Usage: `diffir-batch <input directory> -o <output directory> <options>`
where the `<options>` are those shown above.
