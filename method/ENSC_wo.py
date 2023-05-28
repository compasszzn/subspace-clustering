import warnings
import numpy as np

from sklearn.cluster import SpectralClustering


import progressbar
import spams
from sklearn.decomposition import sparse_encode
from scipy import sparse

from sklearn.preprocessing import normalize
def active_support_elastic_net(X, y, alpha, tau=1.0, algorithm='spams', support_init='knn',
                               support_size=100, maxiter=40):
    """An active support based algorithm for solving the elastic net optimization problem
        min_{c} tau ||c||_1 + (1-tau)/2 ||c||_2^2 + alpha / 2 ||y - c X ||_2^2.

    Parameters
    -----------
    X : array-like, shape (n_samples, n_features)
    y : array-like, shape (1, n_features)
    alpha : float
    tau : float, default 1.0
    algorithm : string, default ``spams``
        Algorithm for computing solving the subproblems. Either lasso_lars or lasso_cd or spams
        (installation of spams package is required).
        Note: ``lasso_lars`` and ``lasso_cd`` only support tau = 1.
    support_init: string, default ``knn``
        This determines how the active support is initialized.
        It can be either ``knn`` or ``L2``.
    support_size: int, default 100
        This determines the size of the working set.
        A small support_size decreases the runtime per iteration while increase the number of iterations.
    maxiter: int default 40
        Termination condition for active support update.

    Returns
    -------
    c : shape n_samples
        The optimal solution to the optimization problem.
	"""
    n_samples = X.shape[0]

    if n_samples <= support_size:  # skip active support search for small scale data
        supp = np.arange(n_samples, dtype=int)  # this results in the following iteration to converge in 1 iteration
    else:
        if support_init == 'L2':
            L2sol = np.linalg.solve(np.identity(y.shape[1]) * alpha + np.dot(X.T, X), y.T)
            c0 = np.dot(X, L2sol)[:, 0]
            supp = np.argpartition(-np.abs(c0), support_size)[0:support_size]
        elif support_init == 'knn':
            supp = np.argpartition(-np.abs(np.dot(y, X.T)[0]), support_size)[0:support_size]

    curr_obj = float("inf")
    for _ in range(maxiter):
        Xs = X[supp, :]
        if algorithm == 'spams':
            cs = spams.lasso(np.asfortranarray(y.T), D=np.asfortranarray(Xs.T),
                             lambda1=tau * alpha, lambda2=(1.0 - tau) * alpha)
            cs = np.asarray(cs.todense()).T
        else:
            cs = sparse_encode(y, Xs, algorithm=algorithm, alpha=alpha)

        delta = (y - np.dot(cs, Xs)) / alpha

        obj = tau * np.sum(np.abs(cs[0])) + (1.0 - tau) / 2.0 * np.sum(np.power(cs[0], 2.0)) + alpha / 2.0 * np.sum(
            np.power(delta, 2.0))
        if curr_obj - obj < 1.0e-10 * curr_obj:
            break
        curr_obj = obj

        coherence = np.abs(np.dot(delta, X.T))[0]
        coherence[supp] = 0
        addedsupp = np.nonzero(coherence > tau + 1.0e-10)[0]

        if addedsupp.size == 0:  # converged
            break

        # Find the set of nonzero entries of cs.
        activesupp = supp[np.abs(cs[0]) > 1.0e-10]

        if activesupp.size > 0.8 * support_size:  # this suggests that support_size is too small and needs to be increased
            support_size = min([round(max([activesupp.size, support_size]) * 1.1), n_samples])

        if addedsupp.size + activesupp.size > support_size:
            ord = np.argpartition(-coherence[addedsupp], support_size - activesupp.size)[
                  0:support_size - activesupp.size]
            addedsupp = addedsupp[ord]

        supp = np.concatenate([activesupp, addedsupp])

    c = np.zeros(n_samples)
    c[supp] = cs
    return c


class ENSC_wo:

    def __init__(self, n_clusters, regu_coef=1., n_neighbors=10, ro=0.5, save_affinity=False):
        """

        :param n_clusters: number of clusters
        :param regu_coef: regularization coefficient i.e. labmda
        :param n_neighbors: number of neighbors of knn graph
        :param ro: post-processing parameters
        :param save_affinity: if True, save affinity matrix
        """
        self.n_clusters = n_clusters
        self.regu_coef = regu_coef
        self.n_neighbors = n_neighbors
        self.ro = ro
        self.save_affinity = save_affinity


    def fit(self, X, gamma=50.0, gamma_nz=True, tau=1.0, algorithm='lasso_lars',
                                    active_support=True, active_support_params=None, n_nonzero=50):
        if algorithm in ('lasso_lars', 'lasso_cd') and tau < 1.0 - 1.0e-10:
            warnings.warn('algorithm {} cannot handle tau smaller than 1. Using tau = 1'.format(algorithm))
            tau = 1.0
        if active_support == True and active_support_params == None:
            active_support_params = {}
        n_samples = X.shape[0]
        rows = np.zeros(n_samples * n_nonzero)
        cols = np.zeros(n_samples * n_nonzero)
        vals = np.zeros(n_samples * n_nonzero)
        curr_pos = 0
        for i in progressbar.progressbar(range(n_samples)):
            # for i in range(n_samples):
            #    if i % 1000 == 999:
            #        print('SSC: sparse coding finished {i} in {n_samples}'.format(i=i, n_samples=n_samples))
            y = X[i, :].copy().reshape(1, -1)
            X[i, :] = 0

            if algorithm in ('lasso_lars', 'lasso_cd', 'spams'):
                if gamma_nz == True:
                    coh = np.delete(np.absolute(np.dot(X, y.T)), i)
                    alpha0 = np.amax(coh) / tau  # value for which the solution is zero
                    alpha = alpha0 / gamma
                else:
                    alpha = 1.0 / gamma

                if active_support == True:
                    c = active_support_elastic_net(X, y, alpha, tau, algorithm, **active_support_params)
                else:
                    if algorithm == 'spams':
                        c = spams.lasso(np.asfortranarray(y.T), D=np.asfortranarray(X.T),
                                        lambda1=tau * alpha, lambda2=(1.0 - tau) * alpha)
                        c = np.asarray(c.todense()).T[0]
                    else:
                        c = sparse_encode(y, X, algorithm=algorithm, alpha=alpha)[0]
            else:
                warnings.warn("algorithm {} not found".format(algorithm))

            index = np.flatnonzero(c)
            if index.size > n_nonzero:
                #  warnings.warn("The number of nonzero entries in sparse subspace clustering exceeds n_nonzero")
                index = index[np.argsort(-np.absolute(c[index]))[0:n_nonzero]]
            rows[curr_pos:curr_pos + len(index)] = i
            cols[curr_pos:curr_pos + len(index)] = index
            vals[curr_pos:curr_pos + len(index)] = c[index]
            curr_pos += len(index)

            X[i, :] = y
        a= sparse.csr_matrix((vals, (rows, cols)), shape=(n_samples, n_samples))
        C = normalize(a, 'l2')
        y_pre, C_final = self.post_proC(C, self.n_clusters, 8, 18)#spectral clustering
        return y_pre

    def post_proC(self, C, K, d, alpha):#对C进行spectral clustering
        # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
        C = 0.5 * (np.abs(C) + np.abs(C.T))
        spectral = SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                      assign_labels='discretize')
        spectral.fit(C)
        grp = spectral.fit_predict(C) + 1

        return grp, C

