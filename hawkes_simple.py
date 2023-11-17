from __future__ import division
from helper import *
from point_process import PointProcess


class HawkesSimple(PointProcess):
    """
    Simple hawkes process model.

    Members:
            # ---------------- Constants ----------------------- #
            #                                                    #
            #  N                 int            number of nodes  #
            #  B                 int          number of kernels  #
            #  num_events        int           number of events  #
            #  T                 float       time of last event  #
            #                                                    #
            # ---------------- Event-related ------------------- #
            #                                                    #
            #  events            list of tuples (i, j, t)        #
            #  node_events       dict {(i, j): [t's]}   << N**2  #
            #                                                    #
            # ---------------- Parameters ---------------------- #
            #                                                    #
            #  gamma             array                    N * N  #
            #  xi                array                N * N * B  #
            #                                                    #
            # ---------------- Hyper-parameters ---------------- #
            #                                                    #
            #  prior_gamma       tuple                        2  #
            #  prior_xi          array                        B  #
            #                                                    #
            # ---------------- Cached statistics --------------- #
            #                                                    #
            #  delta     {(u, v): array(B * num_events(u, v)))}  #
            #  Delta             array                N * N * B  #
            #                                                    #
            # -------------------------------------------------- #
    """
    def __init__(self, num_nodes, events, end_time):
        """
        Initialize parameters for the Hawkes process.

        Args:
            num_nodes: An integer, the number of nodes.
            events: A list containing 3-tuples: [(i, j, t)], where (i, j, t)
                indicates that an edge from i to j appeared at time t.
                Assume that time points start at 0.
        """
        super(HawkesSimple, self).__init__(num_nodes, events, end_time)

        # Model parameters
        self.gamma = np.zeros((self.N, self.N))
        self.xi = np.zeros((self.N, self.N, self.B))

        # Hyper-prior parameters (shared across all params)
        # Gamma shape and scale parameters for base rate and each kernel
        self.priors = np.hstack((np.ones((1 + self.B, 1)),
                                 np.ones((1 + self.B, 1))))
        self.num_params = 2 * (1 + self.B)

        # # Posteriors (shared across all params)
        # self.post_gamma = (0, 0)
        # self.post_xi = np.zeros(self.B)

        # Sufficient statistics
        self.delta = None
        self.Delta = None
        self.suff_stats_cached = False  # Flag
        self.update_suff_stats()

        # Cache log-likelihood computations for efficiency
        # self.loglik_cached = np.zeros((self.N, self.N), dtype=bool)  # False
        # self.loglik_values = np.zeros((self.N, self.N)) + np.nan

        return

    # ----------------------------------------------------------------------- #
    # Intensity/likelihood related functions
    # ----------------------------------------------------------------------- #

    def update_suff_stats(self):
        """
        Pre-compute the sufficient statistics for the event times:

        \delta^{v,u}_{b,i} = \sum_{k: t^{v,u}_k < t^{u,v}_i}
            \phi_b(t^{u,v}_i - t^{v,u}_k)

        \Delta^{v,u}_{b,T} = \sum_k (\Phi_b(T - t^{v,u}_k) - \Phi_b(0))

        and updates self.delta and self.Delta.

        Returns:
            self.delta, dict of arrays.
            self.Delta, array.
        """
        self.delta = {(u, v): np.zeros((self.B, self.num_node_events(u, v)))
                      for u in range(self.N) for v in range(self.N)}
        self.Delta = np.zeros((self.N, self.N, self.B))
        for u in range(self.N):
            for v in range(self.N):
                times = self.get_node_events(u, v)
                recip_times = np.array(self.get_node_events(v, u))

                # Compute \delta^{(v,u)}_{b,i}
                for i, t in enumerate(times):
                    dt = t - recip_times[recip_times < t]  # np.array
                    recip_sum = np.sum(self.kernels(dt), axis=1)  # B * 1
                    self.delta[(u, v)][:, i] = recip_sum

                # Compute \Delta^{(v,u)}_{b,T}
                dT = self.T - recip_times[recip_times < self.T]  # np.array
                recip_sum = np.sum(self.kernels(dT, type='cdf'), axis=1) - \
                    len(dT) * np.sum(self.kernels(0, type='cdf'), axis=1)
                self.Delta[u, v, :] = recip_sum  # B * 1

        self.suff_stats_cached = True

        return self.delta, self.Delta

    def base_rate(self, u, v):
        """
        Computes the base rate \gamma_{pq} n_p n_q for given 1 <= u, v <= N.

        Args:
            u, v: Integers specifying the node indices.

        Returns:
            A float, the computed base rate value.
        """
        return self.gamma[u, v]

    def intensity(self, u, v, times):
        """
        Computes the intensity function \lambda_{pq} for given 1 <= u, v <= N
        evaluated at each time point in times.

        Args:
            u, v: Integers specifying the node indices.
            times: List of time points to be evaluated on or a single number.

        Returns:
            An np.array containing the values of \lambda_{pq} evaluated at each
            time point in times. If times is a single number, then return the
            intensity value (float) evaluated at that time point.
        """
        assert u in range(self.N) and v in range(self.N)
        if isinstance(times, float) or isinstance(times, int):
            # Return intensity value at a single time point
            return self.intensity(u, v, [times])[0]

        lambdas = np.zeros(len(times)) + self.base_rate(u, v)
        # Reciprocal component
        recip_times = np.array(self.get_node_events(v, u))
        for i, t in enumerate(times):
            dt = t - recip_times[recip_times < t]  # np.array
            recip_sum = np.sum(self.kernels(dt), axis=1)  # B * 1
            lambdas[i] += np.dot(self.xi[u, v], recip_sum)

        return lambdas

    def intensity_fast(self, u, v):
        """
        Computes the intensity function \lambda_{pq} for given 1 <= u, v <= N
        evaluated at each time point in self.get_node_events(u, v) using the
        cached sufficient statistics self.delta.
        NOTE: self.delta[(u, v)] is of dimension B * n_{pq}.

        Args:
            u, v: Integers specifying the node indices.

        Returns:
            An np.array containing the values of \lambda_{pq} evaluated at each
            time point in self.node_events[(u, v)].
        """
        assert u in range(self.N) and v in range(self.N)
        if not self.suff_stats_cached:
            self.update_suff_stats()

        lambdas = self.base_rate(u, v)
        lambdas += np.dot(self.xi[u, v], self.delta[(u, v)])

        assert_equal(len(lambdas), self.num_node_events(u, v))

        return lambdas

    def integrated_intensity(self, u, v, T=None):
        """
        Computes the value of the integrated intensity function
        \Lambda_{pq}(0, t) for given 1 <= u, v <= N.

        Args:
            u, v: Integers specifying the node indices.
            T: Float, until time T. Default is self.T.

        Returns:
            Float, value of the integrated intensity function.
        """
        if T is None:
            T = self.T
        else:
            assert T >= 0

        recip_times = np.array(self.get_node_events(v, u))
        dT = T - recip_times[recip_times < T]  # np.array

        recip_sum = np.sum(self.kernels(dT, type='cdf'), axis=1)  # B * 1
        recip_sum -= len(dT) * np.sum(self.kernels(0, type='cdf'), axis=1)

        return self.base_rate(u, v) * T + np.dot(self.xi[u, v], recip_sum)

    def integrated_intensity_fast(self, u, v):
        """
        Computes the value of the integrated intensity function
        \Lambda_{pq}(0, self.T) for given 1 <= u, v <= N using cached
        sufficient statistics self.Delta.
        NOTE: self.delta[(u, v)] is of dimension B * n_{pq}.

        Args:
            u, v: Integers specifying the node indices.

        Returns:
            Float, value of the integrated intensity function.
        """
        if not self.suff_stats_cached:
            self.update_suff_stats()

        temp = self.base_rate(u, v) * self.T
        temp += np.dot(self.xi[u, v], self.Delta[(u, v)])

        return temp

    def predict_probs(self, t0, delta):
        """
        Computes the predicted probability that a link from u to v appears in
        [t, t + delta) based only on the events data from [0, t)
        for all combinations of u and v.
        """
        N = self.N
        t1 = t0 + delta

        prob_dict = np.zeros((N, N))   # Predicted probs that link exists

        for u in range(N):
            for v in range(N):
                recip_times = [t for t in self.get_node_events(v, u) if t < t0]
                recip_times = np.array(recip_times)
                temp0 = self.kernels(t0 - recip_times, type='cdf')
                temp1 = self.kernels(t0 + delta - recip_times, type='cdf')
                recip_sum = np.sum(temp1 - temp0, axis=1)  # B * 1

                Lambda = self.gamma[u, v] * delta
                Lambda += np.dot(self.xi[u, v], recip_sum)

                prob_dict[u, v] = 1. - np.exp(-Lambda)

        return prob_dict

    def predict_receiver(self, u, t):
        """
        Predicts the recipient probs for a message sent from u at time t.
        """
        vals = [self.intensity(u, v, t) for v in range(self.N)]
        vals[u] == 0

        probs = normalize(vals)

        return probs

    # ----------------------------------------------------------------------- #
    # MLE & MCMC
    # ----------------------------------------------------------------------- #

    def loglik(self, gamma, xi):
        """
        Computes the log-likelihood function with all parameters tied.
        """
        if not self.suff_stats_cached:
            self.update_suff_stats()

        loglik = 0
        for u in range(self.N):
            for v in range(self.N):
                gterm = -gamma * self.T
                bterm = -np.dot(xi, self.Delta[u, v])
                lterm = np.sum(np.log(gamma +
                                      np.dot(xi, self.delta[(u, v)][:, i]))
                               for i in range(self.num_node_events(u, v)))
                loglik += gterm + bterm + lterm

        # for u in range(self.N):
        #     for v in range(self.N):
        #         temp = -self.integrated_intensity_fast(u, v)
        #         temp += np.sum(np.log(self.intensity_fast(u, v)))
        #         loglik += temp

        # loglik -= gamma + np.sum(xi)  # Prior

        return loglik

    def loglik_grad(self, gamma, xi):
        """
        Compute the gradient evaluated at gamma, xi.
        """
        if not self.suff_stats_cached:
            self.update_suff_stats()

        # Unroll
        gamma = np.array(gamma)
        if len(gamma.shape) == 0:
            gamma = gamma[np.newaxis]

        if len(gamma.shape) == 1:  # 0-D or 1-D array
            gamma = gamma[:, np.newaxis]

        num = gamma.shape[0]

        gradient = np.zeros((len(gamma), 1 + self.B))
        for u in range(self.N):
            for v in range(self.N):
                Delta = np.insert(self.Delta[u, v], 0, self.T)
                Delta = np.tile(Delta, (num, 1))
                for i in range(self.num_node_events(u, v)):
                    delta = np.insert(self.delta[(u, v)][:, i], 0, 1.)
                    delta = np.tile(delta, (num, 1))
                    denom = gamma + np.dot(xi, np.reshape(self.delta[(u, v)][:, i], (self.B, 1)))
                    denom = np.tile(denom, (1, 1+self.B))
                    gradient += delta/denom

                gradient -= Delta

        # gradient -= 1.  # Prior

        return gradient[0] if len(gradient) == 1 else gradient

    def loglik_hess(self, gamma, xi):
        """
        Compute the Hessian matrix evaluated at gamma, xi.
        """
        if not self.suff_stats_cached:
            self.update_suff_stats()

        hessian = np.zeros((1 + self.B, 1 + self.B))
        for u in range(self.N):
            for v in range(self.N):
                for i in range(self.num_node_events(u, v)):
                    delta = np.insert(self.delta[(u, v)][:, i], 0, 1.)
                    temp = np.outer(delta, delta)
                    denom = (gamma + np.dot(xi, self.delta[(u, v)][:, i]))**2
                    hessian -= temp/denom

        return hessian

    def mle(self, method='grad-ascent', **kwargs):
        """
        Computes the MLE with all parameters tied.
        """
        def neg_loglik_obj(x):
            """
            Computes the negative log-likelihood value.

            Args:
                x: array; x[0] := gamma and x[1:] := xi[:-1]
            """
            gamma, xi = self.unpack_params(x)

            return -self.loglik(gamma, xi)

        def neg_loglik_obj_grad(x):
            """
            Computes the negative log-likelihood value.

            Args:
                x: array; x[0] := gamma and x[1:] := xi[:-1]
            """
            gamma, xi = self.unpack_params(x)

            return -self.loglik_grad(gamma, xi)

        if method == 'grad-ascent':
            bounds = zip([_EPS] + [_EPS] * self.B,
                         [None] + [None] * self.B)  # 1-_EPS

            gamma_init = rand.uniform()
            xi_init = rand.uniform(size=self.B)
            x_init = np.hstack((gamma_init, xi_init))

            res = minimize(neg_loglik_obj,
                           jac=neg_loglik_obj_grad,
                           x0=x_init,
                           method='L-BFGS-B', bounds=bounds, **kwargs)

            assert res.success, "MLE optimization failed ..."
            x = res.x if res.success else None

        elif method == 'coord-ascent':
            x, _, _ = coord_descent(obj_fun=neg_loglik_obj,
                                    num_params=1+n, **kwargs)

        else:
            print("MLE %s method not understood!" % method)

        mle_params = self.unpack_params(x)
        self.set_mle_params(mle_params)

        return mle_params

    # ----------------------------------------------------------------------- #
    # Book-keeping
    # ----------------------------------------------------------------------- #

    def unpack_params(self, x):
        """
        Args:
            x: array; x[0] := gamma and x[1:] := xi[:-1]
        """
        assert_equal(len(x), 1+self.B)

        gamma = x[0]
        xi = x[1:]

        assert_ge(gamma, 0)
        assert all_pos(xi)

        return gamma, xi

    def set_mle_params(self, res):
        """
        Given an array containing the unpacked parameter set their values
            accordingly.
        """
        gamma, xi = res

        self.gamma[:] = gamma
        self.xi[:] = xi

        return

    # ----------------------------------------------------------------------- #
    # Variational inference
    # ----------------------------------------------------------------------- #

    def elbo_mc(self, params, num_mc_iters=200):
        """
        Computes the evidence lower bound for all pairs of nodes, assuming
        tied parameters with the priors
            gamma ~ Gamma(priors[0][:])
            xi[b] ~ Gamma(priors[b][:]) for b = 1, ..., B,
        and posteriors
            gamma ~ Gamma(pvec[0], qvec[0])
            xi[b] ~ Gamma(pvec[b], qvec[b]) for b = 1, ..., B,
        by evalutaing the intergal in the expected log-likelihood using Monte
        Carlo.

        Args:
            pvec: array of length B+1 containing the posterior shape params
                for the base rate and each kernel;
            pvec: array of length B+1 containing the posterior scale params
                for the base rate and each kernel.

        NOTE:
            The implementation supports vectorized computation. Hence,
            pvec and qvec can be 2-D arrays of shape (*, B+1), and the returned
            ELBO value will be a 1-D array of length *.
        """
        if not self.suff_stats_cached:
            self.update_suff_stats()

        # Unroll
        params = np.array(params)
        if len(params.shape) == 1:  # 0-D or 1-D array
            params = params[np.newaxis]

        assert_equal(params.dtype, float)
        assert_equal(params.shape[1], 2 * (1+self.B))
        assert all_pos(params), params

        pvec = params[:, :(1+self.B)]  # Shape params
        qvec = params[:, (1+self.B):]  # Scale params

        # Monte Carlo estimate of log-likelihood
        logliks = np.zeros((params.shape[0], num_mc_iters))
        for k in range(num_mc_iters):  # Monte Carlo iteration
            if (k+1) % 20 == 0:
                print("Computing Monte Carlo estimate: %d / %d ..." % \
                    (k+1, num_mc_iters))

            # xi = rand.lognormal(mean=pvec, sigma=qvec)  # Including gamma
            xi = rand.gamma(shape=pvec, scale=1/qvec)  # Including gamma
            temp = 0
            for u in range(self.N):
                for v in range(self.N):
                    Delta = np.insert(self.Delta[u, v], 0, self.T)  # (1+B)-dim
                    temp -= np.dot(xi, Delta)
                    for i in range(self.num_node_events(u, v)):
                        delta = np.insert(self.delta[(u, v)][:, i], 0, 1.)
                        temp += np.log(np.dot(xi, delta))

            logliks[:, k] = temp

        exloglik = np.mean(logliks, axis=1)  # Monte Carlo average
        print("Estimated expected loglik = %s, std.dev = %s" % \
            (exloglik, np.std(logliks, axis=1)))

        # KL-divergence terms
        kl_terms = kl_gamma(pvec, qvec,
                            np.tile(self.priors[:, 0], (pvec.shape[0], 1)),
                            np.tile(self.priors[:, 1], (pvec.shape[0], 1)))

        kl_sum = np.sum(kl_terms, axis=1)

        res = exloglik - kl_sum

        return res[0] if len(res) == 1 else res

    def elbo(self, params):
        """
        Computes the evidence lower bound for all pairs of nodes, assuming
        tied parameters with the priors
            gamma ~ Gamma(priors[0][:])
            xi[b] ~ Gamma(priors[b][:]) for b = 1, ..., B,
        and posteriors
            gamma ~ Gamma(pvec[0], qvec[0])
            xi[b] ~ Gamma(pvec[b], qvec[b]) for b = 1, ..., B.

        Args:
            pvec: array of length B+1 containing the posterior shape params
                for the base rate and each kernel;
            pvec: array of length B+1 containing the posterior scale params
                for the base rate and each kernel.

        NOTE:
            The implementation supports vectorized computation. Hence,
            pvec and qvec can be 2-D arrays of shape (*, B+1), and the returned
            ELBO value will be a 1-D array of length *.
        """
        if not self.suff_stats_cached:
            self.update_suff_stats()

        # Unroll
        params = np.array(params)
        if len(params.shape) == 1:  # 0-D or 1-D array
            params = params[np.newaxis]

        assert_equal(params.dtype, float)
        assert_equal(params.shape[1], 2 * (1+self.B))
        assert all_pos(params), params

        pvec = params[:, :(1+self.B)]  # Shape params
        qvec = params[:, (1+self.B):]  # Scale params

        # Expected log-likelihood
        exloglik = 0.
        for u in range(self.N):
            for v in range(self.N):
                Delta = np.insert(self.Delta[u, v], 0, self.T)  # (1+B)-dim
                term = -np.dot(pvec/qvec, Delta)

                lterm = 0.
                for i in range(self.num_node_events(u, v)):
                    delta = np.insert(self.delta[(u, v)][:, i], 0, 1.)
                    temp = np.exp(digamma(pvec) - np.log(qvec))
                    lterm += np.log(np.dot(temp, delta))

                exloglik += term + lterm  # Expected log-likelihood

        # KL-divergence terms
        kl_terms = kl_gamma(pvec, qvec,
                            np.tile(self.priors[:, 0], (pvec.shape[0], 1)),
                            np.tile(self.priors[:, 1], (qvec.shape[0], 1)))

        kl_sum = np.sum(kl_terms, axis=1)

        res = exloglik - kl_sum

        return res[0] if len(res) == 1 else res

    def coord_ascent(self, monte_carlo=False, **kwargs):
        """
        Performs coordinate ascent to maximize the evidence lower bound.

        Returns:
            x: array of length 2 * (1+B), converged parameter values.
            x_vals: array of shape (1+max_iter, 2 * (1+B)), stores previous
                params values after each full coordinate descent iteration.
            obj_vals: array of length (1+max_iter), stores previous objective
                values after each full coordinate descent iteration.
        """
        if not self.suff_stats_cached:
            self.update_suff_stats()

        elbo = self.elbo_mc if monte_carlo else self.elbo

        return coord_ascent(obj_fun=elbo, num_params=self.num_params, **kwargs)

    # ----------------------------------------------------------------------- #
    # MCMC
    # ----------------------------------------------------------------------- #

    def metropolis(self, num_samples=1000, burnin=500):
        """
        Metropolis-Hastings sampling to infer gamma and xi.
        """
        def log_exponential_pdf(x, l):
            """
            Log pdf for the Exp(l) distribution evaluated at x.
            """
            return np.log(l) - l * x

        def llik_func(x):
            gamma, xi = self.unpack_params(x)
            return self.loglik(gamma, xi)

        res = np.zeros((num_samples+1, 1+self.B))
        res[0] = rand.normal(loc=.1, scale=.02, size=(1+self.B))  # Initialize
        # res[0] = rand.exponential(size=(1+self.B))  # Initialize

        for i in range(1, num_samples+1):
            if i > 0 and i % 50 == 0:
                print("M-H sampled %d samples ..." % i)

            x_old = res[i-1]
            x_new = rand.normal(loc=x_old, scale=.02)  # Proposal
            # x_new = rand.exponential(scale=1./x_old)  # Proposal

            # # Acceptance ratio
            # temp = llik_func(x_new) - llik_func(x_old)
            # temp += np.sum(log_exponential_pdf(x_old, x_new))
            # temp -= np.sum(log_exponential_pdf(x_new, x_old))
            # ratio = np.exp(min(0, temp))
            ratio = np.exp(min(0, llik_func(x_new) - llik_func(x_old))) \
                if np.all(x_new > 0) else 0

            # print x_old, x_new, ratio

            res[i] = x_new if rand.uniform() < ratio else x_old

        return res[(burnin+1):]

    def slice_sample(self, num_samples=1000):
        """
        Slice sampling to infer gamma and xi.
        """
        def llik_func(x):
            gamma, xi = self.unpack_params(x)
            return self.loglik(gamma, xi)

        res = np.zeros((num_samples+1, 1+self.B))
        res[0] = rand.uniform(size=(1+self.B))  # Initialize
        for i in range(1, num_samples+1):
            if i > 0 and i % 50 == 0:
                print("Slice-sampled %d samples ..." % i)

            res[i] = multivariate_slice_sample(
                x_init=res[i-1],
                ll_func=llik_func, window_size=1, L_bound=_EPS)

        return res[1:]

    # ----------------------------------------------------------------------- #
    # Simulation
    # ----------------------------------------------------------------------- #

    def set_params(self, num_nodes=None, events=None, end_time=None,
                   gamma=None, xi=None):
        """
        Manually set all (or a subset of) the parameters for the Hawkes-IRM.

        Args:
            See self.__init__() description.
        """
        if num_nodes is not None:
            self.N = num_nodes

        if events is not None:
            self.node_events = dict()
            self.num_events = 0
            self.process_node_events(events)
            self.T = max(flatten(self.node_events.values())) \
                if self.node_events else 0
            self.update_suff_stats()

        if end_time is not None:
            assert_ge(end_time, self.T)
            self.T = end_time

        if gamma is not None:
            self.gamma = gamma

        if xi is not None:
            self.xi = xi

        return

    def simulate_single(self, c):
        """
        Simulate a single 1-d self-exciting Hawkes process with intensity
        \lambda_{cc}(t).

        Args:
            c: Integer, node index (in 0, ..., self.N).

        Returns:
            A list of simulated event times.
        """
        assert c in range(self.N)
        self.node_events[(c, c)] = list()  # Clear relevant events history

        num = 1  # Number of simulated events
        rate = self.base_rate(c, c)  # Maximum intensity
        jump_size = float(np.dot(self.xi[c, c], self.kernels(0)))

        # First event
        u = rand.uniform()
        s = -np.log(u) / rate
        if s > self.T:
            return self.node_events[(c, c)]

        self.node_events[(c, c)].append(s)
        # Intensity function is left-continuous
        rate = self.intensity(c, c, s) + jump_size

        # General routine
        while s < self.T:
            u = rand.uniform()
            s += -np.log(u) / rate
            if s >= self.T:
                break

            # Rejection test
            d = rand.uniform()
            new_rate = self.intensity(c, c, s)
            if d <= new_rate / rate:
                self.node_events[(c, c)].append(s)
                # Intensity function is left-continuous
                rate = new_rate + jump_size

                num += 1
            else:
                rate = new_rate

        assert num == len(self.node_events[(c, c)])
        print("Simulated %d events for node %d in [0, %.2f)." % \
            (num, c, self.T))

        return self.node_events[(c, c)]

    def simulate_pair(self, u, v):
        """
        Simulate a pair of 1-d self-exciting Hawkes process with intensity
        \lambda_{pq}(t) and \lambda_{qp}(t).
        The implementation can be generalized to the multi-variate case.

        Args:
            u, v: Integer, node indices (in 0, ..., self.N).

        Returns:
            A list of simualted event times.
        """
        assert u in range(self.N) and v in range(self.N)
        assert u < v  # simulate_pair should only be called once for each pair

        # Clear relevant events history
        self.node_events[(u, v)] = list()
        self.node_events[(v, u)] = list()

        # Intensity functipn
        num = 1  # Total number of simulated events
        rate = self.base_rate(u, v) + self.base_rate(v, u)

        # First event
        U = rand.uniform()
        s = -np.log(U) / rate
        if s > self.T:  # Done
            return self.node_events[(u, v)], self.node_events[(v, u)]

        def attribution_test(t):
            """
            Determines which process the newly generated event t should be
            attributed to.
            """
            r_pq = self.intensity(u, v, t)
            r_qp = self.intensity(v, u, t)
            rate = r_pq + r_qp
            idx = (u, v) if rand.uniform() < r_pq / rate else (v, u)
            return idx, rate

        # Attribution test: which process gets the event
        idx, rate = attribution_test(s)
        self.node_events[idx].append(s)
        # Intensity function is left-continuous
        rate += float(np.dot(self.xi[idx], self.kernels(0)))

        # General routine
        while s < self.T:
            U = rand.uniform()
            s += -np.log(U) / rate
            if s >= self.T:
                break  # Done

            # Rejection test
            d = rand.uniform()
            new_rate = self.intensity(u, v, s) + self.intensity(v, u, s)
            if d <= new_rate / rate:
                idx, rate = attribution_test(s)
                self.node_events[idx].append(s)
                # Intensity function is left-continuous
                rate += float(np.dot(self.xi[idx], self.kernels(0)))
                num += 1
            else:
                rate = new_rate

        assert num == len(self.node_events[(u, v)]) + \
            len(self.node_events[(v, u)])

        print("Simulated %d events for (%d, %d) node pair in [0, %.2f)." % \
            (num, u, v, self.T))

        return self.node_events[(u, v)], self.node_events[(v, u)]

    def simulate(self):
        """
        Simulate event times for all Hawkes processes.
        """
        self.events = list()
        self.node_events = dict()  # Clear relevant events history
        self.suff_stats_cached = False

        # Simulate node-level events (self.node_events)
        for c in range(self.N):
            self.simulate_single(c)

        for u in range(self.N):
            for v in range(u + 1, self.N):
                self.simulate_pair(u, v)

        events = self.extract_events()

        return events
