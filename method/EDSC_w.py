import numpy as np
from scipy.sparse.linalg import svds
from sklearn.cluster import SpectralClustering
from sklearn.preprocessing import normalize


class EDSC_w:

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
        X_ = np.transpose(X)  # shape: n_dim * n_samples 转置X才是真正的X
        X_embedding = X_
        I = np.eye(X.shape[0])
        inv = np.linalg.inv(np.dot(np.transpose(X_embedding), X_embedding) + self.regu_coef * I)#inv是求逆
        C = np.dot(np.dot(inv, np.transpose(X_embedding)), X_)
        Coef = self.thrC(C,  self.ro)  #Efficient Dense Subspace Clustering
        y_pre, C_final = self.post_proC(Coef, self.n_clusters, 8, 18)#spectral clustering

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


    def post_proC(self, C, K, d, alpha):#对C进行spectral clustering
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

