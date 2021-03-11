DiffIR
=========================================
DiffIR is a tool for visually 'diffing' the difference between two sets of rankings. Given a pair of TREC runs containing rankings for multiple queries, DiffIR identifies contrasting queries that have "substantially" different results between the two systems and generates a visual side-by-side comparison illustrating how the key rankings differ.

DiffIR supports multiple strategies for ranking comparison including unsupervised ranking correlations like TauAP and supervised comparison based on existing judgments and ranking metrics. DiffIR additionally accepts term importance weights in order to highlight the terms most relevant to a model's relevance prediction.

TODO:
- Usage
- Expected input formats
- Screenshots
- Fix colab link
- Explanation of term importance options
- Explanation of query contrast options

Want to give it a try? `Get started with a Notebook. <https://colab.research.google.com>`_ |Colab Badge|

Looking for the code? `Find DiffIR on GitHub. <https://github.com/capreolus-ir/diffir>`_

.. |Colab Badge| image:: https://colab.research.google.com/assets/colab-badge.svg
    :alt: Open in Colab
    :scale: 100%
    :target: https://colab.research.google.com



