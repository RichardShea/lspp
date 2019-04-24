from __future__ import division
from helper import *


class PointProcess(object):
    """
    Genral point process model.

    Members:
            # ---------------- Event-related ------------------- #
            #                                                    #
            #  events            list of tuples (i, j, t)        #
            #  node_events       dict {(i, j): [t's]}   << N**2  #
            #                                                    #
            # -------------------------------------------------- #
    """
    def __init__(self, num_nodes, events, end_time=None):
        """
        Initialize parameters for the Hawkes process.

        Args:
            num_nodes: An integer, the number of nodes.
            events: A list containing 3-tuples: [(i, j, t)], where (i, j, t)
                indicates that an edge from i to j appeared at time t.
                Assume that time points start at 0.
            nodes: Optionally provide initial nodeing partition
                (list of lists).

        """
        # Initialize nodes
        self.N = num_nodes  # Number of nodes
        self.B = len(self.kernels(dt=0))  # Number of kernels

        # Load events
        self.events = deepcopy(events)
        self.num_events = 0
        self.node_events = dict()  # A dict mapping key (i, j), where
        #  0 <= i, j < self.N to a list of time points for events from i to j.
        self.process_node_events(self.events)  # Node-level event dict
        # NOTE: self.node_events only contain those (i, j) pairs with which
        #   there are events associated. Therefore, any query should first
        #   check if (i, j) in self.node_events.keys() before proceeding or
        #   call via self.node_events.get(key, []) instead of indexing.
        self.T = max(flatten(self.node_events.values())) if self.events else 0.
        if end_time is not None:
            assert_le(self.T, end_time)
            self.T = end_time

        return

    # ----------------------------------------------------------------------- #
    # Node event operations
    # ----------------------------------------------------------------------- #

    def add_node_event(self, i, j, t):
        """
        Helper function to add an event of type (i, j, t) to self.node_events.

        Args:
            i: source node.
            j: receiver node.
            t: timestamp.
        """
        times = self.node_events.get((i, j), None)
        if times:  # List times exist
            times.append(t)
        else:  # First occurrence; create new list
            self.node_events[(i, j)] = [t]

        return

    def process_node_events(self, events):
        """
        Processes input [(i, j, t)] to node-level event dict {(i, j): [t's]}.

        NOTE: Since edges are sparse, the keys of the output node_events only
        contain those (i, j) pairs with which there are events associated.

        Args:
            events: See Hawkes.__init__() description.
        """
        assert all([0 <= i < self.N and 0 <= j < self.N and t >= 0
                    for (i, j, t) in events])

        if self.node_events:  # Not empty
            warnings.warn("process_node_events does not clear event history!")

        for i, j, t in events:
            self.add_node_event(i, j, t)

        assert_equal(len((flatten(self.node_events.values()))), len(events))

        self.num_events += len(events)

        return

    def extract_events(self):
        """
        Extracts the events from self.node_events to a list of 3-tuples.

        Returns:
            events: list of 3-tuples, [(i, j, t)].
        """
        events = list()
        for (i, j), times in self.node_events.iteritems():
            events.extend([(i, j, t) for t in times])

        assert_equal(len((flatten(self.node_events.values()))), len(events))

        self.events = events
        self.num_events = len(events)

        return events

    def get_node_events(self, i, j):
        """
        Helper function to return the node_events from i to j.
        """
        return self.node_events.get((i, j), list())

    def num_node_events(self, i, j):
        """
        Helper function to return the number of node_events from i to j.
        """
        return len(self.get_node_events(i, j))

    def get_adj_mat(self, t0, delta):
        """
        Computes the binary adjacency matrix indicating whether there exists
        a link from u to v during [t0, t0 + delta) for each pair u, v.
        """
        assert t0 < self.T

        mat = np.zeros((self.N, self.N))
        for u in range(self.N):
            for v in range(self.N):
                times = [t for t in self.get_node_events(u, v)
                         if t >= t0 and t < t0 + delta]
                mat[u, v] = 1 if times else 0

        return mat

    # ----------------------------------------------------------------------- #
    # Intensity/likelihood related functions
    # ----------------------------------------------------------------------- #

    def kernels(self, dt, type='pdf'):
        """
        Evaluates the values of the basis kernels \phi's with time dt.

        Args:
            dt: float/np.array, input time(s).

        Returns:
            List of np.array, containing the value of each kernel function
                evaluated with time t.
        """
        dt = np.array(dt)

        res = np.vstack((
            # gamma_kernel(dt, a=5, type=type),
            # exponential_kernel(dt, tau=.01, type=type),
            # exponential_kernel(dt, tau=.1, type=type),
            exponential_kernel(dt, tau=1/24., type=type),
            exponential_kernel(dt, tau=1., type=type),
            # exponential_kernel(dt, tau=5, type=type),
            exponential_kernel(dt, tau=7., type=type),
            local_periodic_kernel(dt, p=7., l=7., type=type)
            # exponential_kernel(dt, tau=100, type=type)
            ))

        return res

    def intensity(self):
        return

    def integrated_intensity(self):
        return

    def loglik(self):
        return

    # ----------------------------------------------------------------------- #
    # Evaluation
    # ----------------------------------------------------------------------- #

    def homogeneous_poisson(self):
        """
        Estimate the MLE of a homogeneous Poisson intensity parameter.

        Returns:
            A N * N np.array of homogeneous Poisson intensity estimates.
        """
        lambdas = np.zeros((self.N, self.N))  # N * N intensities
        for p in range(self.N):
            for q in range(self.N):
                lambdas[p, q] = len(self.get_node_events(p, q)) / self.T

        return lambdas

    def goodness_of_fit(self):
        """
        Tests the goodness-of-fit of the Hawkes process.

        Under random time change (Theorem 7.4.I of Daley and Vere-Jones), the
        random time transformation \tau = \Lambda(t) = \int_0^t \lambda(u) du
        takes the point process with intensity \lambda(t) into a unit-rate
        Poisson process.

        Returns:
            A dict mapping each node-pair to its transformed event-times.
        """
        transformed_times = dict()
        for (p, q), times in self.node_events.iteritems():
            transformed_times[(p, q)] = [
                self.integrated_intensity(p, q, t) for t in times]

        assert all(t >= 0 for t in flatten(transformed_times.values()))

        return transformed_times
