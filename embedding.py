"""
Various functions for evaluating embeddings (plotting, etc).
"""
from __future__ import division
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.ticker import FormatStrFormatter

_LW = 2   # Line widths
_MW = 2   # Marker line widths


# Heat maps
def plot_dist_heatmap(pp, fname='', _sorted=None):
    """
    Plot heatmap of the estimated node similarity matrices.
    """
    distmat = pp.dist_mat

    # Set diagonal values to be zero
    # for b in range(1+pp.B):
    #     np.fill_diagonal(distmat[b], 0.)

    if _sorted == 'degs':
        # Sort indices based on degrees
        cnt = Counter(i for (i, j, t) in pp.events)
        cnt += Counter(j for (i, j, t) in pp.events)
        inds = [v for v, count in cnt.most_common()]
        distmat = distmat[:, inds, :][:, :, inds]

    # Plot heatmaps for each b diemnsion
    for b in range(1+pp.B):
        if _sorted is not None and _sorted != 'degs':
            if _sorted == 'dist':  # Sort indices based on distance
                degs = np.sum(distmat[b], axis=0)  # Based on zvec
                inds = np.argsort(degs)[::-1]

            elif _sorted == 'clust':
                # Sort indices via hierarchical clustering
                inds = sns.clustermap(distmat[b]).dendrogram_row.reordered_ind

            else:
                raise NameError('Sorting method %s not recognized' % _sorted)

            distmat_b = distmat[b, inds, :][:, inds]

        else:
            distmat_b = distmat[b, :, :]

        plt.figure(figsize=(8, 8))
        sns.heatmap(distmat_b, norm=LogNorm(), cbar=False,
                    xticklabels=False, yticklabels=False)
        plt.savefig('distmat-%s-%d.pdf' % (fname, b), bbox_inches='tight')
        plt.close('all')

    return


def plot_ternary(pp, fname=''):
    """
    Tenary plots.
    """
    xvec = pp.xvec
    num_pts = 1001
    kernel_idx = [0, 1, 2]   # Kernel indices
    for v in range(pp.N):  # for each node v
        dists = np.exp(-np.sum((xvec - np.tile(xvec[v], (pp.N, 1, 1)))**2, axis=2))

        # dist_norms = np.linalg.norm(dists, axis=1)
        dist_norms = np.sqrt(np.sum((xvec - np.tile(xvec[v], (pp.N, 1, 1)))**2, axis=(1, 2)))

        # Plot setup
        # sns.set_style('whitegrid')
        figure = plt.figure(figsize=(8, 12))  # (8, 12)
        gs = gridspec.GridSpec(3, 1, height_ratios=[4, 1, 1])
        gs.update(hspace=0.25)  # Spacing between axes

        # ----------- Simplex plot ----------- #

        temp = dists[:, kernel_idx]
        temp[:, 2] += dists[:, 3]  # Add-on periodic kernel
        row_sums = np.sum(temp, axis=1).reshape((pp.N, 1))
        probs = temp / np.tile(row_sums, (1, len(kernel_idx)))

        ax1 = plt.subplot(gs[0])
        tax = ternary.TernaryAxesSubplot(ax=ax1, scale=1)
        # fig.set_size_inches(8, 7)
        tax.left_axis_label(r"$\phi_1$", fontsize=20)
        tax.right_axis_label(r"$\phi_2$", fontsize=20)
        tax.bottom_axis_label(r"$\phi_3\ +\ \phi_4$", fontsize=20)
        # tax.bottom_axis_label(r"$\phi_4$", fontsize=15)
        tax.boundary(linewidth=2.0)
        tax.gridlines(multiple=.1, color="blue")
        # tax.scatter(probs, marker='o', s=50, facecolors='none',
        #             linewidths=_MW, edgecolors=_green)
        tax.scatter(probs, marker='o', s=50, color='red')
        tax.ticks(axis='lbr', linewidth=1, multiple=.1)
        tax.clear_matplotlib_ticks()
        tax._redraw_labels()

        # ----------- Intensity plot ----------- #

        # sns.set_style('darkgrid')

        ax2 = plt.subplot(gs[1])

        x = np.linspace(0, pp.T, num_pts)
        kernel_ints = np.zeros((pp.N, num_pts, 1+pp.B))
        for u in range(pp.N):
            kernel_ints[u, :, :] = pp.kernel_intensity(v, u, x)

        kernel_ints = np.sum(kernel_ints, axis=0)

        # # Normalize intensities
        # kernel_ints_max = np.max(kernel_ints, axis=0)
        # kernel_ints = kernel_ints / np.tile(kernel_ints_max, (num_pts, 1))

        assert kernel_ints.shape[1] == 1+pp.B
        for b in range(1, 1+pp.B):
            ax2.plot(x, kernel_ints[:, b], color=_COLORS[b-1],
                     label=r'$\phi_%d$' % b)

        # ax2.set_aspect(120)
        ax2.set_xlabel("Time (Days)", fontsize=15)
        ax2.set_ylabel("Intensity", fontsize=15)
        ax2.locator_params(nbins=4, axis='y')
        # ax2.set_ylabel("Normalized Mean Intensity", fontsize=15)
        ax2.legend(loc='upper center', bbox_to_anchor=(.5, 1.4),  # 1.35
                   ncol=4, fancybox=True, shadow=True, fontsize=20)

        # ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2e'))

        # ----------- Histogram of distances ----------- #

        ax3 = plt.subplot(gs[2])
        ax3.hist(dist_norms, 50)
        ax3.set_xlabel("Euclidean Distance to Other Nodes", fontsize=15)
        ax3.set_ylabel("Count", fontsize=15)
        ax3.locator_params(nbins=4, axis='y')

        plt.savefig("ternary-%s-%d.pdf" % (fname, v), bbox_inches='tight')
        plt.close('all')

    return


def roc(fname, methods, num_nodes, num_dims=2, nreps=50):
    """
    Given that there was a link from u to a node at time t, predict which node
      was the receiver.
    """
    N, events_test_dict = stratify_events_test(fname=fname, num_nodes=num_nodes)

    rand.seed(123)

    for method in methods:  # 'LatentSpace'

        try:
            pp = pckl_read("%s-%s-N%dD%d-prop0.7.pp.res" %
                           (fname, method, N, num_dims))
        except IOError:
            print("No such file! %s-%s-N%dD%d-prop0.7.pp.res" % \
                (fname, method, N, num_dims))

            continue

        assert_equal(N, pp.N)

        if 'Embedding' in method:

            Xvec = pp.xvec if method == 'HawkesEmbedding2' else pp.zvec
            embedding_roc(num_nodes=pp.N, events_train=pp.events,
                          events_test=events_test_dict, xvec=Xvec,
                          method=method + '-zvec', fname=fname)

            if method == 'HawkesEmbedding' or method == 'HawkesEmbedding1':
                # Plot reciprocal embeddings
                xvec = pp.xvec.reshape((pp.N, pp.B * pp.D))
                for b in range(pp.B):
                    xvec = pp.xvec[:, b, :]
                    embedding_roc(num_nodes=pp.N, events_train=pp.events,
                                  events_test=events_test_dict, xvec=xvec,
                                  method=method + '-xvec%d' % b, fname=fname)

                # Take the mean of the xvec's
                # xvec = np.mean(pp.xvec, axis=1)
                # embedding_roc(num_nodes=pp.N, events_train=pp.events,
                #               events_test=events_test_dict, xvec=xvec,
                #               method=method + '-xvec', fname=fname)

        elif method == 'LatentSpace':
            embedding_roc(num_nodes=pp.N, events_train=pp.events,
                          events_test=events_test_dict, xvec=pp.xvec,
                          method='LatentSpace', fname=fname)

    return


def embedding_auc(X, W, method='distance', normed=True):
    N = X.shape[0]
    D = X.shape[1]

    assert_equal(X.shape[0], W.shape[0])

    if normed:
        row_norms = np.linalg.norm(X, axis=1).reshape((N, 1))
        X = X / np.tile(row_norms, (1, D))  # Row-normalized

    res = np.zeros((N, N))  # Predicted edge probability
    for i in range(N):
        for j in range(N):
            if method == 'distance':
                res[i, j] = np.exp(-np.sum((X[i] - X[j])**2))
            elif method == 'dotprod':
                res[i, j] = sigmoid(np.dot(X[i], X[j]))
            else:
                raise NameError('Embedding method not recognized!')

        res[i, i] = 0  # No self-loops

    W_true = binarize(W)

    fpr, tpr, thresholds = roc_curve(W_true.flatten(), res.flatten())
    # plt.plot(fpr, tpr)
    # plt.show()

    return fpr, tpr, auc(fpr, tpr)


def project2D(X):
    """
    Project X to 2D column space using PCA.
    """
    pca = PCA(n_components=2)
    pca.fit(X)

    return pca.fit_transform(X)


def embedding_roc(num_nodes, events_train, events_test, xvec, method,
                  _method='distance', _normed=False, fname=''):
    """
    Plot the ROC curve and scatter-plots for the learned embedding xvec
        along with Laplacian eigenmaps for comparison.
    """
    if isinstance(events_test, list):
        events_test = {0: events_test}  # Reduction

    assert isinstance(events_test, dict)

    N = xvec.shape[0]  # Number of nodes
    k = xvec.shape[1]  # Number of latent dims

    assert_equal(N, num_nodes)

    # Spectral clustering
    W_train = events_to_adj(num_nodes, events_train)
    # W_train = binarize(W_train)
    X_spec = spectral_cluster(W_train, k)

    # node2vec
    try:
        n2vec = node2vec_embedding(fname, num_nodes=N, num_dims=k)
    except:
        pass

    for _type, events_test_temp in events_test.items():
        W_test = events_to_adj(num_nodes, events_test_temp)

        assert_equal(N, W_test.shape[0])

        fpr_spec, tpr_spec, auc_spec = embedding_auc(
            X_spec, W_test, method=_method, normed=_normed)

        try:
            fpr_n2v, tpr_n2v, auc_n2v = embedding_auc(
                n2vec, W_test, method=_method, normed=_normed)
        except:
            pass

        fpr_pp, tpr_pp, auc_pp = embedding_auc(
            xvec, W_test, method=_method, normed=_normed)

        try:
            print("Type %d: Spectral: %.4f  node2vec: %.4f  %s: %.4f" % \
                (_type, auc_spec, auc_n2v, method, auc_pp))
        except:
            print("Type %d: Spectral: %.4f  %s: %.4f" % \
                (_type, auc_spec, method, auc_pp))

        # Plot ROC curve
        plt.figure(figsize=(8, 6))
        plt.plot(fpr_pp, tpr_pp, lw=_LW, color=_RED,
                 label=method)
        plt.plot(fpr_spec, tpr_spec, ls='--', lw=_LW, color=_BLUE,
                 label="Spectral")
        try:
            plt.plot(fpr_n2v, tpr_n2v, ls='-.', lw=_LW, color=_PURPLE,
                     label="node2vec")
        except:
            pass
        plt.xlabel("False Positive Rate", fontsize=20)
        plt.ylabel("True Positive Rate", fontsize=20)
        plt.legend(loc=0, fontsize=15)
        plt.savefig('%s-%s-N%d-roc-%d.pdf' % (fname, method, N, _type), bbox_inches='tight')
        plt.close('all')

        if _type == 0:  # Plot scatter plot
            # Project onto 2D plane
            xvec_2d = project2D(xvec)
            xspec_2d = project2D(X_spec)

            plt.figure(figsize=(8, 6))
            plt.scatter(xvec_2d[:, 0], xvec_2d[:, 1], marker='o', s=50, facecolors='none',
                        linewidths=_MW, edgecolors=_BLUE,
                        label=method)
            plt.scatter(xspec_2d[:, 0], xspec_2d[:, 1], marker='v', s=50, facecolors='none',
                        linewidths=_MW, edgecolors=_RED,
                        label="Spectral")
            try:
                n2vec_2d = project2D(n2vec)
                plt.scatter(n2vec_2d[:, 0], n2vec_2d[:, 1], marker='v', s=50, facecolors='none',
                            linewidths=_MW, edgecolors=_PURPLE,
                            label="node2vec")
            except:
                pass

            plt.xlabel("Principal Component 1", fontsize=20)
            plt.ylabel("Principal Component 2", fontsize=20)
            plt.legend(loc=1, fontsize=15)
            plt.savefig('%s-%s-N%d-scatter.pdf' % (fname, method, N), bbox_inches='tight')
            plt.close('all')

    return auc_spec, auc_pp


def pred_acc(fname, methods, num_nodes, num_dims=100, nreps=50, delta=14):
    """
    Given that there was a link from u to a node at time t, predict which node
      was the receiver.
    """
    if isinstance(methods, str):  # Single method
        methods = [methods]

    N, events_test_dict = stratify_events_test(fname=fname, num_nodes=num_nodes)

    # if isinstance(events_test_dict, list):
    #     events_test_dict = {0: events_test_dict}

    # assert isinstance(events_test_dict, dict)

    rand.seed(123)

    events0 = events_test_dict[1]
    events1 = events_test_dict[2]
    events2 = events_test_dict[3]
    events3 = events_test_dict[4]

    events_test = events0 + events1 + events2 + events3
    events_test_idx = [1]*len(events0) + [2]*len(events1) + \
        [3]*len(events2) + [4]*len(events3)

    events_test_sorted = sorted(zip(events_test, events_test_idx),
                                key=lambda e: e[0][2])

    events_test = map(itemgetter(0), events_test_sorted)
    events_test_idx = map(itemgetter(1), events_test_sorted)

    num_test = len(events_test_idx)

    # Randomly sample indices
    idx_list = list()
    for _type in range(1, 5):  # 5 types
        temp = [i for i in range(num_test) if events_test_idx[i] == _type]
        idx = sorted(list(rand.choice(temp, min(nreps, len(temp)),
                                      replace=False)))
        idx_list += idx

    idx_list = [0] + sorted(idx_list)

    # Record results
    res = {method: {_type: list() for _type in range(5)}
           for method in methods}  # ACC, MRR

    f = open("receiver-%s.txt" % fname, 'w')

    f.write("# %s\n" % fname.upper())

    # global res
    for method in methods:
        proc = PointProcess(num_nodes, events_test)

        try:  # Load results
            pp_test = pckl_read("%s-%s-N%dD%d-prop0.7.pp.res" %
                                (fname, method, num_nodes, num_dims))
        except IOError:
            print("File does not exist! %s-%s-N%dD%d-prop0.7.pp.res" % \
                (fname, method, num_nodes, num_dims))

        res_m = res[method]

        for k, idx in enumerate(idx_list[1:]):
            if k % 20 == 0:
                print(method, k)

            i0 = idx_list[k-1] if k > 1 else 0

            # Add events history
            for e in events_test[i0:idx]:
                pp_test.add_node_event(*e)

            pp_test.suff_stats_cached = False
            pp_test.update_suff_stats()

            u, v, t = events_test[idx]  # Current event
            probs = pp_test.predict_receiver(u, t)

            tidx = events_test_idx[idx]

            # Prediction accuracy
            probs_dict = dict(zip(range(pp_test.N), probs))
            acc = 1. * (most_common(probs_dict) == v)

            # Mean reciprocal rank
            num_gr = np.sum(probs > probs[v])
            rank = num_gr + 1
            mrr = 1. / rank

            # Temporal sliding-window link prediction
            t0 = events_test[idx][-1]

            adj_prob = pp_test.predict_probs(t0, delta)
            # adj_prob = pp_test.dist_mat[0, :, :]  # Only homophily LS
            # adj_prob = pp_test.dist_mat[1, :, :]  # Only first kernel LS
            # adj_prob = np.sum(pp_test.dist_mat, axis=0)  # Sum across kernels
            # adj_prob = np.sum(pp_test.dist_mat, axis=0)  # Sum across kernels

            adj_true = proc.get_adj_mat(t0, delta)

            fpr, tpr, thresholds = roc_curve(y_true=adj_true.flatten(),
                                             y_score=adj_prob.flatten())
            auc_score = auc(fpr, tpr)

            adj_true_all = adj_true.flatten()
            adj_prob_all = adj_prob.flatten()
            ind = np.argsort(adj_prob_all)[::-1]
            # prec5 = sum(adj_true_all[ind[:5]]) / 5.
            prec10 = np.sum(adj_true_all[ind[:10]]) / 10.

            # Book-keeping
            res_m[tidx].append((acc, mrr, auc_score, prec10))
            res_m[0].append((acc, mrr, auc_score, prec10))

        for _type in range(5):
            res_arr = np.array(res_m[_type])

            res_acc = res_arr[:, 0]
            res_mrr = res_arr[:, 1]
            res_auc = res_arr[:, 2]
            res_prec = res_arr[:, 3]

            line = "%s Type %d: %d \n" % (method, _type, len(res_acc))
            line += "    acc_mean = %.4f, acc_sd = %.4f;" % \
                (np.mean(res_acc), np.std(res_acc))
            line += " mrr_mean = %.4f, mmr_sd = %.4f\n" % \
                (np.mean(res_mrr), np.std(res_mrr))
            line += "    auc_mean = %.4f, auc_sd = %.4f;" % \
                (np.mean(res_auc), np.std(res_auc))
            line += " prec@10_mean = %.4f, prec@10_sd = %.4f" % \
                (np.mean(res_prec), np.std(res_prec))

            print(line)
            line += '\n'
            f.write(line)

    f.close()

    # return res
    return


def temporal_auc(fname, methods, num_nodes, num_dims=2, delta=14, type=None):
    """
    Compute the AUC scores evaluated at auc_times w/ window-size delta.
    """
    N, events_train, _, T0, T1 = train_test('enron', num_nodes, .7)

    auc_props = np.linspace(.2, .6, 5)
    auc_times = (T0 + T1) * np.linspace(.2, .6, 5)
    assert np.all(auc_times < T0)

    proc = PointProcess(num_nodes, events_train)
    auc_scores = np.zeros_like(auc_times, dtype=np.float64)
    precision_scores = np.zeros_like(auc_times, dtype=np.float64)

    f = open("temporalAUC-%s.txt" % fname, 'w')

    f.write("# %s\n" % fname.upper())

    for method in methods:

        for i, t0 in enumerate(auc_times):

            events_temp = [(u, v, t) for (u, v, t) in events_train if t < t0]

            # pp, params = learn_pp(num_nodes, events_temp, end_time=t0,
            #                       num_dims=num_dims, method=method)

            try:
                pp = pckl_read("%s-%s-N%dD%d-prop%s.pp.res" %
                               (fname, method, N, num_dims, auc_props[i]))

                params = pckl_read("%s-%s-N%dD%d-prop%s.pp.res" %
                                   (fname, method, N, num_dims, auc_props[i]))
            except IOError:
                print("No such file: %s-%s-N%dD%d-prop%s.pp.res" % \
                    (fname, method, N, num_dims, auc_props[i]))
                continue

            adj_prob = pp.predict_probs(t0, delta)
            adj_true = proc.get_adj_mat(t0, delta)

            if type == 'balanced':
                rand.seed(123)

                rows, cols = np.nonzero(adj_true)  # Row and column indices
                all_pairs = set(it.product(range(num_nodes), range(num_nodes)))
                pos_pairs = zip(rows, cols)
                neg_pairs = list(all_pairs - set(pos_pairs))

                neg_idx = rand.choice(len(neg_pairs), len(pos_pairs))
                pairs = [neg_pairs[j] for j in neg_idx] + list(pos_pairs)  # Chosen

                y_true = [adj_true[u, v] for u, v in pairs]
                y_score = [adj_prob[u, v] for u, v in pairs]

                fpr, tpr, thresholds = roc_curve(y_true=adj_true.flatten(),
                                                 y_score=adj_prob.flatten())
            else:
                fpr, tpr, thresholds = roc_curve(y_true=adj_true.flatten(),
                                                 y_score=adj_prob.flatten())

            auc_score = auc(fpr, tpr)
            auc_scores[i] = auc_score

            adj_true_all = adj_true.flatten()
            adj_prob_all = adj_prob.flatten()
            ind = np.argsort(adj_prob_all)[::-1]
            prec5 = sum(adj_true_all[ind[:5]]) / 5.
            prec10 = sum(adj_true_all[ind[:10]]) / 10.

            print("%s T0 = %.4f: auc = %.4f, prec@5 = %.4f, prec@10 = %.4f\n" % \
                (method, t0, auc_score, prec5, prec10))

    f.close()

    # return auc_scores
    return


def test_llik(fname, methods, num_nodes, num_dims=2, nreps=50):
    rand.seed(123)

    print("# %s " % fname.upper())

    N, events_train, events_test, T0, T1 = \
        train_test(fname, num_nodes, .7)

    for method in methods:
        try:
            pp = pckl_read("%s-%s-N%dD%d-prop0.7.pp.res" %
                           (fname, method, N, num_dims))

            params = pckl_read("%s-%s-N%dD%d-prop0.7.mle.res" %
                               (fname, method, num_nodes, num_dims))
        except IOError:
            print("No such file! %s-%s-N%dD%d-prop0.7.pp.res" % \
                (fname, method, N, num_dims))

            continue

        if method == 'HawkesSimple':
            pp_test = HawkesSimple(num_nodes=num_nodes, events=events_test,
                                   end_time=T1)

        elif 'Embedding1' in method:
            pp_test = HawkesEmbedding1(num_nodes=num_nodes, events=events_test,
                                       end_time=T1, num_dims=num_dims)

        elif 'Embedding2' in method:
            pp_test = HawkesEmbedding2(num_nodes=num_nodes, events=events_test,
                                       end_time=T1, num_dims=num_dims)

        elif 'Embedding' in method:
            pp_test = HawkesEmbedding(num_nodes=num_nodes, events=events_test,
                                      end_time=T1, num_dims=num_dims)

        elif method == 'LatentSpace':
            pp_test = LatentSpace(num_nodes=num_nodes, events=events_test,
                                  end_time=T1, num_dims=num_dims)

        print("%s: %.4f" % (method, pp_test.loglik(*params)))

    return


def stratify_events_test(fname, num_nodes, train_prop=.7):
    """
    Train on [0, T0) and test on events in [T0, T].
    """
    lscales = np.array([1/24., 1., 7.]) * np.log(20)

    N, events_train, events_test, T0, T1 = train_test(
        fname, num_nodes, train_prop)

    pp_temp = PointProcess(num_nodes=N, events=events_test, end_time=T1)

    events0 = list()  # No events in past 3*7 days
    events1 = list()  # No events in past 3*1 days but exists in past 3*7 days
    events2 = list()  # No events in past 3*1 hours but exists in past 3*1 days
    events3 = list()  # Exist events in past 3*1 hours
    # Filter events based on lag
    for u in range(N):
        for v in range(N):
            times = sorted(pp_temp.get_node_events(u, v))
            for i, t in enumerate(times):
                e = (u, v, t)
                if i == 0:  # First event
                    events0.append(e)
                    continue

                lag = t - times[i-1]  # Time since last event
                if lag >= lscales[2]:
                    events0.append(e)
                elif lag >= lscales[1]:
                    events1.append(e)
                elif lag >= lscales[0]:
                    events2.append(e)
                else:
                    events3.append(e)

    # Sort on times
    events0 = sorted(events0, key=itemgetter(2))
    events1 = sorted(events1, key=itemgetter(2))
    events2 = sorted(events2, key=itemgetter(2))
    events3 = sorted(events3, key=itemgetter(2))

    events_test_dict = dict(zip(range(5),
                            [events_test, events0, events1, events2, events3]))

    print("num_events:", [len(events_test_dict[x]) for x in range(5)])

    return N, events_test_dict


# def learn_pp(num_nodes, events, end_time, num_dims=2, method='HawkesSimple'):
    """
    Learn the MLE parameters of a PointProcess using method and returns the
        PointProcess equipped with the MLE parameters.

    Example:
        learn_pp(N, events_train, end_time=T0, method='HawkesSimple')
    """
    if method == 'HawkesSimple':
        pp = HawkesSimple(num_nodes, events, end_time)

    elif method == 'HawkesEmbedding':
        pp = HawkesEmbedding(num_nodes, events, end_time, num_dims)

    elif method == 'LatentSpace':
        pp = LatentSpace(num_nodes, events, end_time, num_dims)
        pp.C_xvec = 1.
        pp.C_alpha = 1.

    mle_params = pp.mle()  # Unpacked MLE parameters

    pckl_write(pp, "ppmle-%s-N%dD%dT%.4f.res" %
               (method, num_nodes, num_dims, end_time))

    return pp, mle_params


def train_test(filename, num_nodes=None, train_prop=.7):
    """
    Read Enron data and split training-test sets.

    Args:
        num_nodes: only consider the num_nodes largest-degree nodes.
        train_prop: proportion of events in the training set.
    """

    (N, T, events) = pckl_read("../Events/%s-events.pckl" % filename)

    cnt = Counter(i for (i, j, t) in events)
    cnt += Counter(j for (i, j, t) in events)

    if num_nodes is not None:
        assert num_nodes <= N
        N = num_nodes
        # Only consider nodes with largest degrees
        V = set(map(itemgetter(0), cnt.most_common(num_nodes)))
        V_dict = dict(zip(V, range(num_nodes)))
        events = [(V_dict[i], V_dict[j], t)
                  for (i, j, t) in events if i in V and j in V]

    print("Number of nodes: %s" % N)
    print("Total number of events: %s" % len(events))

    events = sorted(events, key=itemgetter(2))  # Sort on times

    # T0 = train_prop * T
    # events_train = [(i, j, t) for (i, j, t) in events if t <= T0]
    # events_test = [(i, j, t) for (i, j, t) in events if t > T0]
    # print "Training-set proportion: %.4f" % (len(events_train) / len(events))

    num_events = len(events)
    ind0 = int(train_prop * num_events)
    events_train = events[:ind0]
    events_test = events[ind0:]
    T0 = events_train[-1][2]

    events_test = [(u, v, t - T0) for (u, v, t) in events_test]

    print("Training-set proportion: %.4f  T0: %.4f" % \
        (len(events_train) / float(num_events), T0))
    print("Test-set proportion: %.4f  T: %.4f" % \
        (len(events_test) / float(num_events), T))

    return N, events_train, events_test, T0, T - T0


def train_valid_test(filename, num_nodes=None):
    """
    Read data and split into training (50%), validation (20%), and
        test (30%) sets.

    Args:
        num_nodes: only consider the num_nodes largest-degree nodes.
        train_prop: proportion of events in the training set.
    """

    (N, T, events) = pckl_read("../Events/%s-events.pckl" % filename)

    cnt = Counter(i for (i, j, t) in events)
    cnt += Counter(j for (i, j, t) in events)

    if num_nodes is not None:
        assert num_nodes <= N
        N = num_nodes
        # Only consider nodes with largest degrees
        V = set(map(itemgetter(0), cnt.most_common(num_nodes)))
        V_dict = dict(zip(V, range(num_nodes)))
        events = [(V_dict[i], V_dict[j], t)
                  for (i, j, t) in events if i in V and j in V]

    print("Number of nodes: %s" % N)
    print("Total number of events: %s" % len(events))

    events = sorted(events, key=itemgetter(2))  # Sort on times

    num_events = len(events)
    ind0 = int(.5 * num_events)
    ind1 = ind0 + int(.2 * num_events)
    events_train = events[:ind0]
    events_valid = events[ind0:ind1]
    events_test = events[ind1:]
    T0 = events_train[-1][2]
    T1 = events_valid[-1][2]

    events_test = [(u, v, t - T0) for (u, v, t) in events_test]

    print("Training-set proportion: %.4f  T0: %.4f" % \
        (len(events_train) / float(num_events), T0))
    print("Validation-set proportion: %.4f  T1: %.4f" % \
        (len(events_valid) / float(num_events), T1))
    print("Test-set proportion: %.4f  T: %.4f" % \
        (len(events_test) / float(num_events), T))

    return events_train, events_valid, events_test, T0, T1 - T0, T - T1


def events_to_adj(num_nodes, events):
    N = num_nodes

    cnt = Counter((i, j) for (i, j, t) in events)
    cnt += Counter((j, i) for (i, j, t) in events)

    temp = np.zeros((len(cnt), 3))
    for k, ((i, j), e) in enumerate(cnt.items()):
        temp[k] = [i, j, e]

    W = coo_matrix((temp[:, 2], (temp[:, 0], temp[:, 1])), shape=(N, N))
    W = W.toarray()
    # print "Graph density: %.6f" % (np.sum(W != 0) / N**2)

    return W


def train_events_to_edge_list(filename, num_nodes, train_prop=.7):
    """
    Read events_train data and write training data to edge_list with weights.

    Args:
        num_nodes: only consider the num_nodes largest-degree nodes.
        train_prop: proportion of events in the training set.
    """

    N, events_train, events_test, T0, T1 = train_test(
        filename, num_nodes, train_prop)

    cnt = Counter((i, j) for (i, j, t) in events_train)  # directed

    edge_list = np.zeros((len(cnt), 3))
    for k, ((i, j), e) in enumerate(cnt.items()):
        edge_list[k] = [i, j, e]

    print("Number of nodes: %s" % N)
    print("Total number of events_train: %s" % len(events_train))

    with open(filename + '-N%d.edgelist' % N, 'w') as f:
        for i, j, e in edge_list:
            f.write("%d %d %f\n" % (i, j, e))

    return edge_list


def node2vec_embedding(filename, num_nodes, num_dims):
    """
    Read the embeddings obtained from node2vec.
    """
    xvec = np.zeros((num_nodes, num_dims))

    temp = np.loadtxt("node2vec/emb/%s-N%d-D%d.emd" %
                      (filename, num_nodes, num_dims), skiprows=1)
    # xvec = temp[np.argsort(temp[:, 0]), 1:]  # Sort on vertices

    for line in temp:
        v = int(line[0])  # node id
        xvec[v] = line[1:]

    return xvec


def compute_node2vec(filenames=['enron', 'purdue-email', 'purdue-fb']):
    """
    Compute node2vec embeddings.
    """
    # python node2vec/src/main.py --input node2vec/graph/enron-N155.edgelist \
    #     --weighted --dimensions 100 --directed --output node2vec/emb/enron-N155-D100.emd

    # python node2vec/src/main.py --input node2vec/graph/purdue-fb-N226.edgelist \
    #     --weighted --dimensions 100 --directed --output node2vec/emb/purdue-fb-N226-D100.emd

    # python node2vec/src/main.py --input node2vec/graph/purdue-email-N488.edgelist \
    #     --weighted --dimensions 100 --directed --output node2vec/emb/purdue-email-N488-D100.emd

    import glob
    import os

    # for fname in filenames:
    #     for num_nodes in [100, None]:
    #         train_events_to_edge_list(fname, num_nodes, .7)

    for input_fname in set(glob.glob("node2vec/graph/*.edgelist")):
        output_fname = input_fname.split('/')[-1].rstrip('.edgelist')
        for num_dims in [2, 128]:
            submit_str = "python node2vec/src/main.py --input %s --weighted \
                          --dimensions %d --directed \
                          --output node2vec/emb/%s-D%d.emd" % \
                          (input_fname, num_dims, output_fname, num_dims)
            print(submit_str)
            os.system(submit_str)

    return


def spectral_cluster(W, k=2):
    N = W.shape[0]

    assert np.all(W.T == W), "Adjacency matrix not symmetric!"
    # if np.all(np.sum(W, 0) > 0):
    #     print "Isolated node exists!"

    degs = np.sqrt(np.sum(W, 0) + _EPS)   # Degree sequence sqrt
    Dinv = np.diag(1. / degs)
    Lsym = np.identity(N) - np.dot(np.dot(Dinv, W), Dinv)

    w, v = np.linalg.eig(Lsym)
    # assert np.all(w >= 0), "Laplacian positive semi-definite!"

    idx = np.argsort(w)
    # idx = [ind for ind in idx if w[ind] > _EPS]  # Threshold eigenvalue
    w = w[idx]  # Sort
    v = v[:, idx]  # Sort

    # assert_ge(len(idx), k, "Laplacian not enough embedding-dimension!")

    X_spec = v[:, 0:k]  # Latent coordinates
    # i0 = np.sum(w < _EPS)
    # X_spec = v[:, i0:(i0+k)]  # Latent coordinates

    # ---------------------------------------------------------------------- #
    # # Random walk Laplacian

    # degs = np.sum(W, 0) + _EPS   # Degree sequence
    # D = np.diag(degs)
    # Dinv = np.diag(1. / degs)
    # Lrw = np.dot(Dinv, D - W)

    # w, v = np.linalg.eig(Lrw)
    # idx = np.argsort(w)
    # w = w[idx]  # Sort
    # v = v[idx]  # Sort

    # assert np.all(w >= 0), "Laplacian positive semi-definite!"
    # i0 = np.sum(w < _EPS)
    # X_spec = v[:, i0:(i0+k)]  # Latent coordinates
    # ---------------------------------------------------------------------- #

    # ---------------------------------------------------------------------- #
    # # Embedding per component

    # X_spec = np.zeros((W.shape[0], k))
    # components = ig.Graph.Adjacency(W.tolist()).components()
    # for v_list in components:
    #     if len(v_list) == 1:  # Isolated node
    #         continue

    #     W_temp = W[v_list, :][:, v_list]

    #     degs = np.sum(W_temp, 0)  # Degree sequence
    #     D = np.diag(degs)
    #     Dinv = np.diag(1. / degs)
    #     Lrw = np.dot(Dinv, D - W_temp)

    #     w, v = np.linalg.eig(Lrw)
    #     idx = np.argsort(w)
    #     w = w[idx]  # Sort
    #     v = v[idx]  # Sort

    #     assert np.all(w > -_EPS), "Laplacian positive semi-definite!"
    #     i0 = np.sum(w < _EPS)
    #     X_spec[v_list, :] = v[:, i0:(i0+k)]  # Latent coordinates
    # ---------------------------------------------------------------------- #

    # row_norms = np.linalg.norm(X_spec, axis=1).reshape((N, 1))
    # X_spec += _EPS  # Smoothing
    # X_spec = X_spec / np.tile(row_norms, (1, k))  # Row-normalized

    # plt.scatter(X_spec[:, 0], X_spec[:, 1])
    # plt.show()

    return X_spec


def ternary_plot(points, filename=''):
    """
    Draw ternary plot.
    """
    fig, tax = ternary.figure(scale=1)

    fig.set_size_inches(8, 7)
    # tax.set_title("Plot", fontsize=20)
    tax.left_axis_label(r"$\phi_1$", fontsize=15)
    tax.right_axis_label(r"$\phi_2$", fontsize=15)
    tax.bottom_axis_label(r"$\phi_4$", fontsize=15)
    tax.boundary(linewidth=2.0)
    tax.gridlines(multiple=.1, color="blue")
    # Plot a few different styles with a legend
    tax.scatter(points, color=_RED)
    tax.legend(fontsize=15)
    tax.ticks(axis='lbr', linewidth=1, multiple=.1)
    tax.clear_matplotlib_ticks()
    tax._redraw_labels()
    fig.savefig('ternary-%s.pdf' % filename, bbox_inches='tight')
    plt.close('all')

    return


def kernel_intensity(pp, u, v, times):
    """
    Compute intensity function for each kernel dimension.
    """
    assert u in range(pp.N) and v in range(pp.N)

    if isinstance(times, float):
        # Return intensity value at a single time point
        return pp.intensity(u, v, [times])[0]

    if not pp.xi_cached:
        pp.update_xi()

    lambdas = np.zeros((len(times), 1+pp.B))
    # Reciprocal component
    recip_times = np.array(pp.get_node_events(v, u))
    for i, t in enumerate(times):
        dt = t - recip_times[recip_times < t]  # np.array
        recip_sum = np.sum(pp.kernels(dt), axis=1)  # B * 1
        lambdas[i, 1:] += pp.xi[u, v] * recip_sum

    lambdas *= pp.beta[u, v]
    # lambdas += pp.base_rate(u, v)
    lambdas[:, 0] = pp.base_rate(u, v)

    return lambdas
