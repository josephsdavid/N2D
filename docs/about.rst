About N2D
=========

**N2D** is a python library implementation of the "deep" clustering method described in this `brilliant paper <https://arxiv.org/abs/1908.05968v5>`_, and by all metrics represents the absolute state of the art in time series/sequence and image clustering. The source code for the software is available `here <https://github.com/josephsdavid/N2D>`_.

In this section we will talk about the motivations for N2D, what it is, and the goals for this package.


What is N2D?
------------------

N2D is short for "Not too deep" clustering. A "not too deep" clustering algorithm works as follows:

1. The data goes into an autoencoder (or other representation learning neural network), which is trained, learning a powerful, concise representation (embedding) of the data.

2. The autoencoded embedding then goes into a manifold learner, in this case primarily UMAP (while t-sne and ISOMAP are also usable), which finds a *local manifold* within the data

3. The local manifold is then sent into a clustering algorithm, which clusters the data


What does it do?
~~~~~~~~~~~~~~~~

The idea of N2D is as follows: by first learning an embedding of the data, and then learning the manifold of the autoencoded data, we transform the data into a form that is readily clusterable, again demonstrated in the `paper <https://arxiv.org/abs/1908.05968v5>`_. N2D is competitive with the most state of the art deep clustering techniques out there, with the benefit of being simple, relatively fast, and intuitive, and represents an excellent path for future research.


Purpose of the library
-----------------------

The purpose of this library is to provide A) an easy library for regular use and B) an extensible framework for future research. 


Citation
--------------

Please cite the original authors of the algorithm if you use N2D in your research. ::

        @article{2019arXiv190805968M,
        title = {N2D:(Not Too) Deep Clustering via Clustering the Local Manifold of an Autoencoded Embedding},
        author = {{McConville}, Ryan and {Santos-Rodriguez}, Raul and {Piechocki}, Robert J and {Craddock}, Ian},
        journal = {arXiv preprint arXiv:1908.05968},
        year = "2019",
        }
