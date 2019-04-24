from __future__ import division
from bisect import bisect
import cPickle
import igraph as ig
import itertools as it
from math import floor, ceil
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import scipy as sp
import os
import pylab
import seaborn as sns
import sys
import warnings
from collections import Counter
from copy import deepcopy
from itertools import chain
from numpy import random as rand
from operator import itemgetter
# from slice_sampler import slice_sample, multivariate_slice_sample
from scipy.sparse import coo_matrix
from scipy.stats import chi2_contingency, probplot
from scipy.stats import gamma as gamma_dist
from scipy.misc import logsumexp
# from scipy.special import gamma as Gamma
from scipy.optimize import minimize, minimize_scalar, check_grad
from scipy.optimize import rosen, rosen_der
from scipy.special import gammaln, digamma, polygamma
from multiprocessing import Pool
from sklearn.metrics import roc_curve, auc
# from sklearn.linear_model import LogisticRegression

# warnings.simplefilter("error", RuntimeWarning)
# warnings.simplefilter("ignore", DeprecationWarning)

np.set_printoptions(precision=3, suppress=True)

_EPS = 1e-6
_INF = 1e6

# sns.set_style('whitegrid')
# sns.set_style('ticks')
# sns.set_style({'xtick.direction': 'in', 'ytick.direction': 'in'})


# --------------------------------------------------------------------------- #
# Simple helper functions
# --------------------------------------------------------------------------- #

def assert_equal(a, b, message=""):
    """Check if a and b are equal."""
    assert a == b, "Error: %s != %s ! %s" % (a, b, message)
    return


def assert_le(a, b, message=""):
    """Check if a and b are equal."""
    assert a <= b, "Error: %s > %s ! %s" % (a, b, message)
    return


def assert_ge(a, b, message=""):
    """Check if a and b are equal."""
    assert a >= b, "Error: %s < %s ! %s" % (a, b, message)
    return


def is_close(a, b, rel_tol=1e-09, abs_tol=0.0):
    """Check approximate equality for floating-point numbers."""
    return abs(a-b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)


def is_unique(l):
    """Check if all the elements in list l are unique."""
    assert type(l) is list, "Type %s is not list!" % type(l)
    return len(l) == len(set(l))


def all_pos(l):
    """Check if all the elements in list/array l are positive."""
    return np.all(l > 0)


def rand_choice(d, size=None):
    """Convenience wrapper for rand.choice to draw from a probability dist."""
    assert type(d) is Counter or type(d) is dict
    assert is_close(sum(d.values()), 1.)  # Properly normalized
    return rand.choice(a=d.keys(), p=d.values(), size=size)


def filterNone(l):
    """Returns the list l after filtering out None (but 0"s are kept)."""
    return [e for e in l if e is not None]


def flatten(l, r=1):
    """Flatten a nested list/tuple r times."""
    return l if r == 0 else flatten([e for s in l for e in s], r-1)


def binarize(x):
    assert isinstance(x, np.ndarray)
    return 1 * (x > 0)


def deduplicate(seq):
    """Remove duplicates from list/array while preserving order."""
    seen = set()
    seen_add = seen.add  # For efficiency due to dynamic typing
    return [e for e in seq if not (e in seen or seen_add(e))]


def most_common(cnt):
    """
    Probabilistic version of Counter.most_common() in which the most-common
    element is drawn uniformly at random from the set of most-common elements.
    """
    assert type(cnt) == Counter or type(cnt) == dict

    max_count = max(cnt.values())
    most_common_elements = [i for i in cnt.keys() if cnt[i] == max_count]
    return rand.choice(most_common_elements)


def normalize(counts, alpha=0.):
    """
    Normalize counts to produce a valid probability distribution.

    Args:
        counts: A Counter/dict/np.ndarray/list storing un-normalized counts.
        alpha: Smoothing parameter (alpha = 0: no smoothing;
            alpha = 1: Laplace smoothing).

    Returns:
        A Counter/np.array of normalized probabilites.
    """
    if type(counts) is Counter:
        # Returns the normalized counter without modifying the original one
        temp = sum(counts.values()) + alpha * len(counts.keys())
        dist = Counter({key: (counts[key]+alpha) / temp
                        for key in counts.keys()})
        return dist

    elif type(counts) is dict:
        # Returns the normalized dict without modifying the original one
        temp = sum(counts.values()) + alpha * len(counts.keys())
        dist = {key: (counts[key]+alpha) / temp for key in counts.keys()}
        return dist

    elif type(counts) is np.ndarray:
        temp = sum(counts) + alpha * len(counts)
        dist = (counts+alpha) / temp
        return dist

    elif type(counts) is list:
        return normalize(np.array(counts))

    else:
        raise NameError("Input type %s not understood!" % type(counts))


def accuracy(pred_labels, true_labels, test_vs):
    """
    Computes classification accuracy on test vertices.

    Args:
        pred_labels: A dict of predicted labels for each node.
        true_labels: A dict of ground-truth labels for each node.
        test_vs: A list of test node ids (int) over which the accuracy
            will be measured.

    Returns:
        acc: (int) classification accuracy on test_vs.
    """
    acc = sum([pred_labels[i] == true_labels[i] for i in test_vs])/len(test_vs)

    return acc


def rand_bern(p=.5, size=1):
    """Generate Bernoulli random numbers."""
    return rand.binomial(n=1, p=p, size=size)


def sigmoid(z):
    """Logistic sigmoid function."""
    return 1. / (1 + np.exp(-z))


# --------------------------------------------------------------------------- #
# Kernel functions
# --------------------------------------------------------------------------- #

# Reference: # http://www.cs.toronto.edu/~duvenaud/cookbook/index.html

def exponential_kernel(t, tau, type='pdf'):
    """
    Exponential kernel.

    Args:
        t: Input variable.
        tau: Length scale (square-root).
    """
    if type == 'pdf':
        # return 1/tau * np.exp(-t / tau)
        return np.exp(-t / tau)
    elif type == 'cdf':
        # return 1 - np.exp(-t / tau)
        return tau * (1 - np.exp(-t / tau))
    else:
        raise NameError('Kernel type not understood')


def gamma_kernel(t, a, type='pdf'):
    """
    Gamma repulsive kernel.

    Args:
        t: Input variable.
        tau: Length scale (square-root).
    """
    if type == 'pdf':
        # return rate**shape * (t**(shape-1)) * np.exp(-rate*t) / Gamma(shape)
        return gamma_dist.pdf(t, a=2)
    elif type == 'cdf':
        return gamma_dist.cdf(t, a=2)
    else:
        raise NameError('Kernel type not understood')


def periodic_kernel(t, p, type='pdf'):
    """
    Periodic kernel.

    Args:
        t: Input variable.
        p: Period.
    """
    # return np.exp(-2 * np.sin(np.pi * t / p)**2 / tau)  # GP
    if type == 'pdf':
        return np.sin(np.pi*t / p)**2
    elif type == 'cdf':
        return t/2. - p * np.sin(2*np.pi*t/p) / (4*np.pi)
    else:
        raise NameError('Kernel type not understood')


def local_periodic_kernel(t, p, l, type='pdf'):
    """
    Local periodic kernel with exponential decay.

    Args:
        t: Input variable.
        p: Period.
        l: Length scale (square-root).
    """
    if type == 'pdf':
        return np.exp(-t/l) * (np.sin(np.pi*t/p)**2)
    elif type == 'cdf':
        numer = (p**2) * (np.cos(2*np.pi*t/p) - 1)
        numer -= 2*l*p*np.pi * np.sin(2*np.pi*t/p)
        numer *= np.exp(-t/l) * l
        numer += 4 * (1 - np.exp(-t/l)) * (l**3) * (np.pi**2)
        denom = 2 * (p**2 + 4 * (l**2) * (np.pi**2))
        return numer / denom
    else:
        raise NameError('Kernel type not understood')


# --------------------------------------------------------------------------- #
# Plotting setup
# --------------------------------------------------------------------------- #

font = {'size': 15}
mpl.rc('font', **font)
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15

_RED = '#F8766D'
_GREEN = '#7CAE00'
_BLUE = '#00BFC4'
_PURPLE = '#C77CFF'
_ORANGE = '#FFA500'

_red = '#D7191C'
_orange = '#FDAE61'
_green = '#ABDDA4'
_blue = '#2B83BA'
_brown = '#A6611A'
_gold = '#DFC27D'
_lblue = '#ABD9E9'
_lgreen = '#80CDC1'

# Color wheel
_COLORS = [_BLUE, _RED, _GREEN, _PURPLE, _ORANGE, _blue, _brown, _gold, _lblue,
           _lgreen, _red, _green, 'navy', 'green', 'blue', 'magenta', 'yellow']


def rand_color():
    """Generate a random hex color."""
    return '#%02X%02X%02X' % tuple(rand.randint(low=0, high=255, size=3))


def rand_jitter(arr, scale=.01):
    if len(arr) == 0:
        return arr

    stdev = scale * (np.max(arr) - np.min(arr))
    return arr + rand.normal(size=len(arr)) * stdev


# --------------------------------------------------------------------------- #
# Simple parallelism using Pool
# --------------------------------------------------------------------------- #


def parallel(f, sequence):
    pool = Pool()
    # pool = ThreadPool()
    result = pool.map(f, sequence)
    pool.close()
    pool.join()

    return [res for res in result if res is not None]


# --------------------------------------------------------------------------- #
# I/O
# --------------------------------------------------------------------------- #

def pckl_write(data, filename):
    with open(filename, 'w') as f:
        cPickle.dump(data, f)

    return


def pckl_read(filename):
    with open(filename, 'r') as f:
        data = cPickle.load(f)

    return data
