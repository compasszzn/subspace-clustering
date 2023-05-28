import numpy as np
from scipy.sparse.linalg import svds
from sklearn.cluster import SpectralClustering
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize
from sklearn.metrics import pairwise_kernels


class GCSC_Kernel:

    def __init__(self, n_clusters, regu_coef=1., n_neighbors=10, kernel='rbf', gamma=1., ro=0.5, save_affinity=False):
        """

        :param n_clusters: number of clusters
        :param regu_coef: regularization coefficient i.e. labmda
        :param n_neighbors: number of neighbors of knn graph
        :param kernel: kernel functions, default 'rbf'. See sklearn for available kernels
        :param gamma: only used in rbf kernel
        :param ro: post-processing parameters
        :param save_affinity: if True, save affinity matrix
        """
        self.n_clusters = n_clusters
        self.regu_coef = regu_coef
        self.n_neighbors = n_neighbors
        self.kernel = kernel
        self.gamma = gamma
        self.ro = ro
        self.save_affinity = save_affinity

    def __adjacent_mat(self, x, n_neighbors=10):
        """
        Construct normalized adjacent matrix, N.B. consider only connection of k-nearest graph
        :param x: array like: n_sample * n_feature
        :return:
        """
        A = kneighbors_graph(x, n_neighbors=n_neighbors, include_self=True).toarray()
        A = A * np.transpose(A)
        D = np.diag(np.reshape(np.sum(A, axis=1) ** -0.5, -1))
        normlized_A = np.dot(np.dot(D, A), D)
        return normlized_A

    def fit(self, X):
        A = self.__adjacent_mat(X, self.n_neighbors)
        if self.kernel == 'linear':
            K = pairwise_kernels(X, metric='linear')
        elif self.kernel == 'polynomial':
            K = pairwise_kernels(X, metric='polynomial', gamma=0.05, degree=3)
        elif self.kernel == 'sigmoid':
            K = pairwise_kernels(X, metric='sigmoid', gamma=0.5)
        elif self.kernel == 'rbf':
            K = pairwise_kernels(X, metric='rbf', gamma=self.gamma)
        else:
            raise Exception('Invalid kernel')
        I = np.eye(X.shape[0])

        T = np.dot(np.transpose(A), K)
        #T = K

        inv = np.linalg.inv(np.dot(T, K) + self.regu_coef * I)
        C = np.dot(inv, T)
        Coef = self.thrC(C,  self.ro)
        y_pre, C_final = self.post_proC(Coef, self.n_clusters, 8, 18)

        return y_pre

    def thrC(self, C, ro):
        if ro < 1:
            N = C.shape[1]
            Cp = np.zeros((N, N))
            S = np.abs(np.sort(-np.abs(C), axis=0))
            Ind = np.argsort(-np.abs(C), axis=0)
            for i in range(N):
                cL1 = np.sum(S[:, i]).astype(float)
                stop = False
                csum = 0
                t = 0
                while (stop == False):
                    csum = csum + S[t, i]
                    if csum > ro * cL1:
                        stop = True
                        Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                    t = t + 1
        else:
            Cp = C
        return Cp


    def post_proC(self, C, K, d, alpha):
        # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
        C = 0.5 * (C + C.T)
        r = d * K + 1
        U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
        U = U[:, ::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis=1)
        Z = U.dot(U.T)
        Z = Z * (Z > 0)
        L = np.abs(Z ** alpha)
        L = L / L.max()
        L = 0.5 * (L + L.T)
        spectral = SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                      assign_labels='discretize')
        spectral.fit(L)
        grp = spectral.fit_predict(L) + 1
        return grp, L
