# Latent Space Point Process Models for Dynamic Networks

Python 2.7 code for the paper:

    Decoupling homophily and reciprocity with latent space network models.
    Jiasen Yang, Vinayak Rao, and Jennifer Neville.
    In Proceedings of the 33rd Conference on Uncertainty in Artificial Intelligence (UAI), 2017

#### Description of Files ####

`point_process.py`: Base class for point process models.

`hawkes_simple.py`, `hawkes_simple_test.py`:          Implements the HP model.

`hawkes_embedding.py`, `hawkes_embedding_test.py`:    Implements the DLS model.

`hawkes_embedding2.py`, `hawkes_embedding2_test.py`:  Implements the RLS model.

The PLS and BLS models can be obtained by regularizing the DLS model.

`embedding.py`: Various functions for evaluating embeddings.

`helper.py`: Miscellaneous utility functions.

#### Demo Example ####

Learning the DLS model from simulated data:
    `Hawkes-DLS-demo.ipynb`

Enron data file (in cPickle format):
    `enron-events.pckl`

Please contact [Jiasen Yang](http://www.stat.purdue.edu/~yang768/) for questions or comments.
