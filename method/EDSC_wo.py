import numpy as np
from sklearn.cluster import SpectralClustering


class EDSC_wo:

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


    def fit(self, X):
        print(X.shape)
        X_ = np.transpose(X)  # shape: n_dim * n_samples 转置X才是真正的X
        X_embedding = X_
        I = np.eye(X.shape[0])
        inv = np.linalg.inv(np.dot(np.transpose(X_embedding), X_embedding) + self.regu_coef * I)#inv是求逆
        C = np.dot(np.dot(inv, np.transpose(X_embedding)), X_)
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



