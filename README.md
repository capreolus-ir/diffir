[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Documentation Status](https://readthedocs.org/projects/diffir/badge/?version=latest)](https://capreolus.readthedocs.io/projects/diffir/?badge=latest)
[![PyPI version fury.io](https://badge.fury.io/py/diffir.svg)](https://pypi.python.org/pypi/diffir/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black) 

# DiffIR
DiffIR is a tool for visually 'diffing' the difference between two sets of rankings. Given a pair of TREC runs containing rankings for multiple queries, DiffIR identifies contrasting queries that have "substantially" different results between the two systems and generates a visual side-by-side comparison illustrating how the key rankings differ.

DiffIR supports multiple strategies for ranking comparison including unsupervised ranking correlations like TauAP and supervised comparison based on existing judgments and ranking metrics. DiffIR additionally accepts term importance weights in order to highlight the terms most relevant to a model's relevance prediction.
   

<!-- TODO: -->
<!-- - usage example -->
<!-- - colab notebook -->
<!-- - screenshot of system? -->
