.. _installation:

============
Installation
============

The library can be installed using pip:

Stable version:

.. code-block:: bash

    pip install deslib

Latest version (under development):

.. code-block:: bash

    pip install git+https://github.com/scikit-learn-contrib/DESlib

DESlib is tested to work with Python 3.5, 3.6 and 3.7. The dependency requirements are:

* scipy(>=0.13.3)
* numpy(>=1.10.4)
* scikit-learn(>=0.19.0)

These dependencies are automatically installed using the pip commands above.

Optional dependencies
=====================
To use Faiss (Fair AI Similarity Search), a fast implementation of KNN that can use GPUs, follow the instructions below:
https://github.com/facebookresearch/faiss/blob/master/INSTALL.md

Note that Faiss is only available on Linux and MacOS.