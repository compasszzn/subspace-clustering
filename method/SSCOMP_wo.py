import numpy as np
from sklearn.cluster import SpectralClustering
import progressbar
from scipy import sparse
from sklearn.preprocessing import normalize

class SSCOMP_wo:

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


    def fit(self, X,n_nonzero=10, thr=1.0e-6):
        n_samples = X.shape[0]
        rows = np.zeros(n_samples * n_nonzero, dtype=int)
        cols = np.zeros(n_samples * n_nonzero, dtype=int)
        vals = np.zeros(n_samples * n_nonzero)
        curr_pos = 0

        for i in progressbar.progressbar(range(n_samples)):
            # for i in range(n_samples):
            residual = X[i, :].copy()  # initialize residual
            supp = np.empty(shape=(0), dtype=int)  # initialize support
            residual_norm_thr = np.linalg.norm(X[i, :]) * thr
            for t in range(n_nonzero):  # for each iteration of OMP
                # compute coherence between residuals and X
                coherence = abs(np.matmul(residual, X.T))
                coherence[i] = 0.0
                # update support
                supp = np.append(supp, np.argmax(coherence))
                # compute coefficients
                c = np.linalg.lstsq(X[supp, :].T, X[i, :].T, rcond=None)[0]
                # compute residual
                residual = X[i, :] - np.matmul(c.T, X[supp, :])
                # check termination
                if np.sum(residual ** 2) < residual_norm_thr:
                    break

            rows[curr_pos:curr_pos + len(supp)] = i
            cols[curr_pos:curr_pos + len(supp)] = supp
            vals[curr_pos:curr_pos + len(supp)] = c
            curr_pos += len(supp)

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

