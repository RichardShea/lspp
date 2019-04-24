from __future__ import division
from helper import *
from point_process import PointProcess
from hawkes_simple import HawkesSimple


class HawkesEmbedding2(HawkesSimple):
    """
    Hawkes process embedding model.

    Members:
            # ---------------- Constants ----------------------- #
            #                                                    #
            #  N                 int            number of nodes  #
            #  B                 int          number of kernels  #
            #  num_events        int           number of events  #
            #  T                 float       time of last event  #
            #                                                    #
            # ---------------- Parameters ---------------------- #
            #                                                    #
            #  gamma             array                    N * N  #
            #  xvec              array                N * B * D  #
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
    def __init__(self, num_nodes, events, end_time, num_dims):
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
        HawkesSimple.__init__(self, num_nodes, events, end_time)

        self.D = num_dims  # Number of latent dimensions

        # Model parameters
        self.gamma = np.zeros((self.N, self.N))
        self.beta = np.zeros(self.B)  # Means \xi in the writeup
        self.xvec = np.zeros((self.N, self.D))

        self.xi = None
        self.xi_cached = False
        self.update_xi()

        self.param_lens = [1, self.B, self.N * self.D]
        self.num_params = sum(self.param_lens)

        # Regularization parameters
        self.C_xvec = 0.
        self.C_beta = 0.

        return

    # ----------------------------------------------------------------------- #
    # Intensity/likelihood related functions
    # ----------------------------------------------------------------------- #

    def update_xi(self):
        """
        Update the xi's using the latent embedding coordinates xvec.
        """
        xi = np.zeros((self.N, self.N, self.B))
        for u in range(self.N):
            for v in range(self.N):
                temp = np.exp(-np.linalg.norm(self.xvec[u] - self.xvec[v])**2)
                xi[u, v, :] = self.beta * temp  # (B, )

        self.xi = xi
        self.xi_cached = True

        return self.xi

    @property
    def dist_mat(self):
        """
        Node-similarity matrix.
        """
        res = np.zeros((self.N, self.N))

        for u in range(self.N):
            for v in range(u, self.N):
                temp = self.xvec[u] - self.xvec[v]  # (D, 1)
                sim = np.exp(-np.linalg.norm(temp)**2)
                res[u, v] = res[v, u] = sim

        # np.fill_diagonal(res, 0.)  # Set diagonal values to be zero

        return res

    def base_rate(self, u, v):
        """
        Computes the base rate \gamma_{pq} n_p n_q for given 1 <= u, v <= N.

        Args:
            u, v: Integers specifying the node indices.

        Returns:
            A float, the computed base rate value.
        """

        return self.gamma[u, v]

    def kernel_intensity(self, u, v, times):
        """
        Compute intensity function for each kernel dimension.
        """
        assert u in range(self.N) and v in range(self.N)

        if isinstance(times, float):
            # Return intensity value at a single time point
            return self.intensity(u, v, [times])[0]

        if not self.xi_cached:
            self.update_xi()

        lambdas = np.zeros((len(times), 1+self.B))  # Include base_rate
        # Reciprocal component
        recip_times = np.array(self.get_node_events(v, u))
        for i, t in enumerate(times):
            dt = t - recip_times[recip_times < t]  # np.array
            recip_sum = np.sum(self.kernels(dt), axis=1)  # B * 1
            lambdas[i, 1:] += self.xi[u, v] * recip_sum

        # lambdas += self.base_rate(u, v)
        lambdas[:, 0] = self.base_rate(u, v)

        return lambdas

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
        if isinstance(times, float):
            # Return intensity value at a single time point
            return self.intensity(u, v, [times])[0]

        if not self.xi_cached:
            self.update_xi()

        lambdas = np.zeros(len(times))
        # Reciprocal component
        recip_times = np.array(self.get_node_events(v, u))
        for i, t in enumerate(times):
            dt = t - recip_times[recip_times < t]  # np.array
            recip_sum = np.sum(self.kernels(dt), axis=1)  # B * 1
            lambdas[i] += np.dot(self.xi[u, v], recip_sum)

        lambdas += self.base_rate(u, v)

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

        if not self.xi_cached:
            self.update_xi()

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

        if not self.xi_cached:
            self.update_xi()

        recip_times = np.array(self.get_node_events(v, u))
        dT = T - recip_times[recip_times < T]  # np.array

        recip_sum = np.sum(self.kernels(dT, type='cdf'), axis=1)  # B * 1
        recip_sum -= len(dT) * np.sum(self.kernels(0, type='cdf'), axis=1)

        res = self.base_rate(u, v) * T
        res += np.dot(self.xi[u, v], recip_sum)

        return res

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

        if not self.xi_cached:
            self.update_xi()

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

                Lambda = self.base_rate(u, v) * delta
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
    # MLE
    # ----------------------------------------------------------------------- #

    def loglik(self, gamma, beta, xvec):
        """
        Computes the penalized log-likelihood function.
        """
        if not self.suff_stats_cached:
            self.update_suff_stats()

        if not self.xi_cached:
            self.update_xi()

        assert all_pos(gamma)
        assert all_pos(beta)

        assert_equal(beta.shape, (self.B, ))
        assert_equal(xvec.shape, (self.N, self.D))

        loglik = 0
        for u in range(self.N):
            for v in range(self.N):
                if u == v:  # Skip
                    continue

                xi = beta * np.exp(-np.linalg.norm(xvec[u] - xvec[v])**2)

                gterm = -gamma * self.T
                bterm = -np.dot(xi, self.Delta[u, v])
                lterm = [np.log(gamma + np.dot(xi, self.delta[(u, v)][:, i]) + _EPS)
                         for i in range(self.num_node_events(u, v))]
                loglik += gterm + bterm + sum(lterm)

        # Prior
        # loglik -= gamma

        return loglik

    def loglik_grad(self, gamma, beta, xvec):
        """
        Computes the gradients of the penalized log-likelihood function.
        """
        if not self.suff_stats_cached:
            self.update_suff_stats()

        N = self.N
        B = self.B
        D = self.D

        # ----- Cache computations -----

        sigx = np.zeros((N, N))
        # 1 / (1 + exp(-\|x_u^(b), x_v^(b)\|_2^2)) for each u, v, b
        for u in range(N):
            for v in range(u+1, N):  # v > u
                temp = np.exp(-np.linalg.norm(xvec[u] - xvec[v])**2)
                sigx[u, v] = sigx[v, u] = temp  # Symmetry

        # assert np.array_equal(sigx, sigx.swapaxes(0, 1))

        hsum = np.zeros((N, N))
        # \sum_{i=1}^{n_{uv}} h^{-1}(u, v, i) for each u,v
        rsum = np.zeros((N, N, B))
        # \sum_{i=1}^{n_{uv}} \delta^{uv}_{b,i} h^{-1}(u, v, i) for each u,v,b
        for u in range(N):
            for v in range(N):
                if v == u:
                    continue

                n_uv = self.num_node_events(u, v)
                hinv_uv = np.zeros(n_uv)  # (n_{uv},) h(u, v, i) for each i
                for i in range(n_uv):
                    temp = np.dot(self.delta[u, v][:, i], beta) * sigx[u, v]
                    temp += gamma
                    hinv_uv[i] = 1. / (temp + _EPS)

                hsum[u, v] = np.sum(hinv_uv)

                rsum[u, v] = np.dot(self.delta[u, v], hinv_uv)  # (B,))
                rsum[u, v] -= self.Delta[u, v]

        # Symmetrize
        # rsum = rsum + np.swapaxes(rsum, 0, 1)

        # ----- Compute gradients -----

        grad_gamma = -(self.N**2 - self.N) * self.T + np.sum(hsum)
        grad_beta = np.zeros(B)
        grad_xvec = np.zeros((N, D))

        xbar = np.mean(xvec, axis=1)  # Average across b's

        for v in range(N):
            for u in range(N):
                if u == v:
                    continue

                grad_beta += sigx[u, v] * rsum[u, v]

                temp = np.dot(rsum[u, v] + rsum[v, u], beta)
                temp *= sigx[u, v]

                grad_xvec[v] += temp * 2. * (xvec[u] - xvec[v])

            # Penalty grad
            grad_xvec[v] -= self.C_xvec * xvec[v]

        grad_beta -= self.C_beta * np.sign(beta)

        return grad_gamma, grad_beta, grad_xvec

    def mle(self, method='grad-ascent', _debug=True, **kwargs):
        """
        Computes the MLE.
        """
        def neg_loglik_obj(x):
            """
            Computes the negative log-likelihood value.

            Args:
                x: array; x[0] := gamma and x[1:] := xvec
            """
            loglik = self.loglik(*self.unpack_params(x))

            # Regularization terms
            gamma, beta, xvec = self.unpack_params(x)
            loglik -= self.C_beta * np.sum(np.abs(beta))
            loglik -= self.C_xvec * np.sum(xvec**2) / 2.

            return -loglik

        def neg_loglik_obj_grad(x):
            """
            Computes the negative log-likelihood value.

            Args:
                x: array; x[0] := gamma and x[1:] := xvec
            """
            grad_vec = self.loglik_grad(*self.unpack_params(x))

            x_grad = self.pack_params(*grad_vec)

            return -x_grad

        if _debug:  # Debug mode
            global _Nfeval
            _Nfeval = 1

            def callback_fun(x):
                global _Nfeval
                print _Nfeval, x, neg_loglik_obj(x)
                # print "{0:4d} {1: 3.6f}".format(_Nfeval, neg_loglik_obj(x))
                _Nfeval += 1

                return

        bounds = zip([_EPS] * self.num_params, [None] * self.num_params)

        lower_bounds = [_EPS] * (1 + self.B)
        lower_bounds += [None] * (self.N * self.D)  # xvec
        upper_bounds = [None] * (1 + self.B)
        upper_bounds += [None] * (self.N * self.D)  # xvec
        bounds = zip(lower_bounds, upper_bounds)
        assert_equal(len(bounds), self.num_params)

        gamma_init = rand.uniform()
        beta_init = rand.uniform(size=self.B)
        xvec_init = rand.uniform(size=self.N * self.D)
        x_init = np.hstack((gamma_init, beta_init, xvec_init))

        if method == 'grad-ascent':
            if _debug:
                res = minimize(neg_loglik_obj, x0=x_init,
                               jac=neg_loglik_obj_grad,
                               method='L-BFGS-B', bounds=bounds,
                               callback=callback_fun, **kwargs)
            else:
                res = minimize(neg_loglik_obj, x0=x_init,
                               jac=neg_loglik_obj_grad,
                               method='L-BFGS-B', bounds=bounds, **kwargs)

            if not res.success:
                print "MLE optimization failed ..."
                print res
                # assert res.success

            # x = res.x if res.success else None
            x = res.x

        elif method == 'coord-ascent':
            x, _, _ = coord_descent(obj_fun=neg_loglik_obj,
                                    x_init=x_init, bounds=bounds, **kwargs)

        else:
            print "MLE %s method not understood!" % method

        mle_params = self.unpack_params(x)
        self.set_mle_params(mle_params)
        self.update_xi()

        return mle_params

    # ----------------------------------------------------------------------- #
    # Book-keeping
    # ----------------------------------------------------------------------- #

    def pack_params(self, gamma, beta, xvec):
        """
        Returns:
            x: array;
        """
        N = self.N
        B = self.B
        D = self.D

        assert isinstance(gamma, float), gamma
        assert_equal(beta.shape, (B,))
        assert_equal(xvec.shape, (N, D))

        x = np.concatenate((np.array([gamma]), beta.flatten(), xvec.flatten()))
        assert_equal(x.shape, (self.num_params,))

        return x

    def unpack_params(self, x):
        """
        Args:
            x: array;
        """
        N = self.N
        B = self.B
        D = self.D

        assert_equal(x.shape, (self.num_params, ))
        inds = np.cumsum(self.param_lens, dtype=int)  # Array split points
        gamma, beta, xvec, _ = np.split(x, inds)

        # beta = np.array(beta)
        # beta = np.reshape(beta, (N, N))
        xvec = np.reshape(xvec, (N, D))

        assert all_pos(gamma)
        assert all_pos(beta)

        return gamma, beta, xvec

    def set_mle_params(self, res):
        """
        Given an array containing the unpacked parameter set their values
            accordingly.
        """
        gamma, beta, xvec = res

        self.gamma[:] = gamma
        # self.beta = beta
        self.beta[:] = beta
        self.xvec = xvec

        self.xi_cached = False

        return

    # ----------------------------------------------------------------------- #
    # Simulation
    # ----------------------------------------------------------------------- #

    def set_params(self, num_nodes=None, num_dims=None,
                   events=None, end_time=None,
                   gamma=None, beta=None, xvec=None):
        """
        Manually set all (or a subset of) the parameters for the Hawkes-IRM.

        Args:
            See self.__init__() description.
        """
        if num_nodes is not None:
            self.N = num_nodes

        if num_dims is not None:
            self.D = num_dims

        if events is not None:
            self.node_events = dict()
            self.num_events = 0
            self.process_node_events(events)
            self.T = max(flatten(self.node_events.values())) \
                if self.node_events else 0

        if end_time is not None:
            assert_ge(end_time, self.T)
            self.T = end_time

        if gamma is not None:
            self.gamma = gamma

        if beta is not None:
            self.beta = beta

        if xvec is not None:
            self.xvec = xvec

        self.update_suff_stats()

        return

    def simulate(self):
        """
        Simulate event times for all Hawkes processes.
        """
        self.events = list()
        self.node_events = dict()  # Clear relevant events history
        self.suff_stats_cached = False
        self.xi_cached = False

        # Simulate node-level events (self.node_events)
        for c in range(self.N):
            self.simulate_single(c)

        for u in range(self.N):
            for v in range(u + 1, self.N):
                self.simulate_pair(u, v)

        events = self.extract_events()

        return events
