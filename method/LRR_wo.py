import numpy as np
from sklearn.cluster import SpectralClustering
from tool.solve_lrr import solve_lrr
from sklearn.neighbors import kneighbors_graph
class LRR_wo:

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
    def __adjacent_mat(self, x, n_neighbors=10):
        """
        使用knn算法计算邻接矩阵
        Construct normlized adjacent matrix, N.B. consider only connection of k-nearest graph
        :param x: array like: n_sample * n_feature
        :return:
        """
        A = kneighbors_graph(x, n_neighbors=n_neighbors, include_self=True).toarray()
        A = A * np.transpose(A)
        D = np.diag(np.reshape(np.sum(A, axis=1) ** -0.5, -1))
        normlized_A = np.dot(np.dot(D, A), D)
        return normlized_A


    def fit(self, X):
        Af = self.__adjacent_mat(X, self.n_neighbors)
        X_ = np.transpose(X)  # shape: n_dim * n_samples 转置X才是真正的X
        X_embedding = X_

        A=X_embedding
        #A=np.dot(X_,Af)

        lamb = 0.1
        C, E = solve_lrr(X_, A, lamb)

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
