import numpy as np
from scipy.linalg import orth
import numpy as np


def solve_l1l2(W, lamb):
    n = W.shape[1]
    E = W.copy()
    for i in range(n):
        E[:, i] = solve_l2(W[:, i], lamb)
    return E


def solve_l2(w, lamb):
    nw = np.linalg.norm(w)
    if nw > lamb:
        x = (nw - lamb) * w / float(nw)
    else:
        x = np.zeros((max(w.shape),))
    return x


def exact_alm_lrr_l1v2(D, A, lamb=None, tol=1e-7, maxIter=1000, display=False):
    m, n = D.shape
    k = A.shape[1]
    n_max = max(m, n)
    if not lamb:
        lamb = 1.0 / np.sqrt(n_max)

    maxIter_primal = 10000
    # initialize
    Y = np.sign(D)
    norm_two = np.linalg.norm(Y, 2)
    norm_inf = np.linalg.norm(Y.flatten(1), np.inf) / lamb
    dual_norm = max(norm_two, norm_inf)
    Y /= dual_norm

    W = np.zeros((k, n))

    Z_hat = np.zeros((k, n))
    E_hat = np.zeros((m, n))
    # parameters
    dnorm = np.linalg.norm(D, 'fro')
    tolProj1 = 1e-6 * dnorm

    anorm = np.linalg.norm(A, 2)
    tolProj2 = 1e-6 * dnorm / anorm

    mu = 0.5 / norm_two  # this one can be tuned
    rho = 6              # this one can be tuned

    # pre-computation
    if m >= k:
        inv_ata = np.linalg.inv(np.eye(k) + A.T.dot(A))
    else:
        inv_ata = np.eye(k) - np.linalg.solve((np.eye(m) + A.dot(A.T)).T,
                                              A).T.dot(A)

    iter = 0
    while iter < maxIter:
        iter += 1

        # solve the primal problem by alternative projection
        primal_iter = 0

        while primal_iter < maxIter_primal:
            primal_iter += 1
            temp_Z, temp_E = Z_hat, E_hat

            # update J
            temp = temp_Z + W / mu
            U, S, V = np.linalg.svd(temp, 'econ')
            V = V.T

            diagS = S
            svp = len(np.flatnonzero(diagS > 1.0 / mu))
            diagS = np.maximum(0, diagS - 1.0 / mu)

            if svp < 0.5:  # svp = 0
                svp = 1

            J_hat = U[:, 0:svp].dot(np.diag(diagS[0:svp]).dot(V[:, 0:svp].T))

            # update Z
            temp = J_hat + A.T.dot(D - temp_E) + (A.T.dot(Y) - W) / mu
            Z_hat = inv_ata.dot(temp)

            # update E
            temp = D - A.dot(Z_hat) + Y / mu
            E_hat = np.maximum(0, temp - lamb/mu) + np.minimum(0, temp +
                                                               lamb/mu)

            if np.linalg.norm(E_hat - temp_E, 'fro') < tolProj1 and \
               np.linalg.norm(Z_hat - temp_Z) < tolProj2:
                break

        H1 = D - A.dot(Z_hat) - E_hat
        H2 = Z_hat - J_hat
        Y = Y + mu * H1
        W = W + mu * H2
        mu = rho * mu

        # stop Criterion
        stopCriterion = max(np.linalg.norm(H1, 'fro') / dnorm,
                            np.linalg.norm(H2, 'fro') / dnorm * anorm)
        if display:
            print('LRR: Iteration', iter, '(', primal_iter, '), mu ', mu, \
                  ', |E|_0 ', np.sum(np.abs(E_hat.flatten(1) > 0)), \
                  ', stopCriterion ', stopCriterion)

        if stopCriterion < tol:
            break

    return (Z_hat, E_hat)

def exact_alm_lrr_l21v2(D, A, lamb, tol=1e-7, maxIter=1000, display=False):
    m, n = D.shape
    k = A.shape[1]

    maxIter_primal = 10000
    # initialize
    Y = np.sign(D)
    norm_two = np.linalg.norm(Y, 2)
    norm_inf = np.linalg.norm(Y.flatten(1), np.inf) / lamb

    dual_norm = max(norm_two, norm_inf)
    Y /= dual_norm

    W = np.zeros((k, n))

    Z_hat = np.zeros((k, n))
    E_hat = np.zeros((m, n))
    # parameters
    dnorm = np.linalg.norm(D, 'fro')
    tolProj1 = 1e-6 * dnorm

    anorm = np.linalg.norm(A, 2)
    tolProj2 = 1e-6 * dnorm / anorm

    mu = 0.5 / norm_two  # this one can be tuned
    rho = 6              # this one can be tuned

    # pre-computation
    if m >= k:
        inv_ata = np.linalg.inv(np.eye(k) + A.T.dot(A))
    else:
        inv_ata = np.eye(k) - np.linalg.solve((np.eye(m) + A.dot(A.T)).T,
                                              A).T.dot(A)

    iter = 0
    while iter < maxIter:
        iter += 1

        # solve the primal problem by alternative projection
        primal_iter = 0

        while primal_iter < maxIter_primal:
            primal_iter += 1
            temp_Z, temp_E = Z_hat, E_hat

            # update J
            temp = temp_Z + W / mu
            U, S, V = np.linalg.svd(temp, 'econ')
            V = V.T

            diagS = S
            svp = len(np.flatnonzero(diagS > 1.0 / mu))
            diagS = np.maximum(0, diagS - 1.0 / mu)

            if svp < 0.5:  # svp = 0
                svp = 1

            J_hat = U[:, 0:svp].dot(np.diag(diagS[0:svp]).dot(V[:, 0:svp].T))

            # update Z
            temp = J_hat + A.T.dot(D - temp_E) + (A.T.dot(Y) - W) / mu
            Z_hat = inv_ata.dot(temp)

            # update E
            temp = D - A.dot(Z_hat) + Y / mu
            E_hat = solve_l1l2(temp, lamb / mu)

            if np.linalg.norm(E_hat - temp_E, 'fro') < tolProj1 and \
               np.linalg.norm(Z_hat - temp_Z) < tolProj2:
                break

        H1 = D - A.dot(Z_hat) - E_hat
        H2 = Z_hat - J_hat
        Y = Y + mu * H1
        W = W + mu * H2
        mu = rho * mu

        # stop Criterion
        stopCriterion = max(np.linalg.norm(H1, 'fro') / dnorm,
                            np.linalg.norm(H2, 'fro') / dnorm * anorm)
        if display:
            print('LRR: Iteration', iter, '(', primal_iter, '), mu ', mu, \
                  ', |E|_2,0 ', np.sum(np.sum(E_hat ** 2, 1) > 0), \
                  ', stopCriterion ', stopCriterion)

        if stopCriterion < tol:
            break

    return (Z_hat, E_hat)

def inexact_alm_lrr_l1(X, A, lamb, display=False):
    tol = 1e-8
    maxIter = 1e6
    d, n = X.shape
    m = A.shape[1]
    rho = 1.1
    max_mu = 1e10
    mu = 1e-6
    atx = A.T.dot(X)
    inv_a = np.linalg.inv(A.T.dot(A) + np.eye(m))
    # Initializing optimization variables
    # intialize
    J = np.zeros((m, n))
    Z = np.zeros((m, n))
    E = np.zeros((d, n))  # sparse

    Y1 = np.zeros((d, n))
    Y2 = np.zeros((m, n))
    # Start main loop
    iter = 0
    if display:
        print("initial,rank=%f" % np.linalg.matrix_rank(Z))

    while iter < maxIter:
        iter += 1
        # update J
        temp = Z + Y2 / mu
        U, sigma, V = np.linalg.svd(temp, 'econ')
        V = V.T
        svp = len(np.flatnonzero(sigma > 1.0 / mu))
        if svp >= 1:
            sigma = sigma[0:svp] - 1.0 / mu
        else:
            svp = 1
            sigma = np.array([0])

        J = U[:, 0:svp].dot(np.diag(sigma).dot(V[:, 0:svp].T))
        # udpate Z
        Z = inv_a.dot(atx - A.T.dot(E) + J + (A.T.dot(Y1) - Y2) / mu)
        # update E
        xmaz = X - A.dot(Z)
        temp = xmaz + Y1 / mu
        E = np.maximum(0, temp - lamb / mu) + np.minimum(0, temp + lamb / mu)

        leq1 = xmaz - E
        leq2 = Z - J
        stopC = max(np.max(np.abs(leq1)), np.max(np.abs(leq2)))
        if display and (iter == 1 or np.mod(iter, 50) == 0 or stopC < tol):
            print("iter", iter, ",mu=", mu, ",rank=", \
                  np.linalg.matrix_rank(Z, tol=1e-3*np.linalg.norm(Z, 2)), \
                  ",stopALM=", stopC)

        if stopC < tol:
            break
        else:
            Y1 += mu * leq1
            Y2 += mu * leq2
            mu = min(max_mu, mu * rho)

    return (Z, E)

def inexact_alm_lrr_l21(X, A, lamb, display=False):
    tol = 1e-8
    maxIter = 1e6
    d, n = X.shape
    m = A.shape[1]
    rho = 1.1
    max_mu = 1e10
    mu = 1e-6
    atx = A.T.dot(X)
    inv_a = np.linalg.inv(A.T.dot(A) + np.eye(m))
    # Initializing optimization variables
    # intialize
    J = np.zeros((m, n))
    Z = np.zeros((m, n))
    E = np.zeros((d, n))  # sparse

    Y1 = np.zeros((d, n))
    Y2 = np.zeros((m, n))
    # Start main loop
    iter = 0
    if display:
        print("initial,rank=%f" % np.linalg.matrix_rank(Z))

    while iter < maxIter:
        iter += 1
        # update J
        temp = Z + Y2 / mu
        U, sigma, V = np.linalg.svd(temp, 'econ')
        V = V.T
        svp = len(np.flatnonzero(sigma > 1.0 / mu))
        if svp >= 1:
            sigma = sigma[0:svp] - 1.0 / mu
        else:
            svp = 1
            sigma = np.array([0])

        J = U[:, 0:svp].dot(np.diag(sigma).dot(V[:, 0:svp].T))
        # udpate Z
        Z = inv_a.dot(atx - A.T.dot(E) + J + (A.T.dot(Y1) - Y2) / mu)
        # update E
        xmaz = X - A.dot(Z)
        temp = xmaz + Y1 / mu
        E = solve_l1l2(temp, lamb / mu)

        leq1 = xmaz - E
        leq2 = Z - J
        stopC = max(np.max(np.abs(leq1)), np.max(np.abs(leq2)))
        if display and (iter == 1 or np.mod(iter, 50) == 0 or stopC < tol):
            print("iter", iter, ",mu=", mu, ",rank=", \
                  np.linalg.matrix_rank(Z, tol=1e-3*np.linalg.norm(Z, 2)), \
                  ",stopALM=", stopC)

        if stopC < tol:
            break
        else:
            Y1 += mu * leq1
            Y2 += mu * leq2
            mu = min(max_mu, mu * rho)

    return (Z, E)



def solve_lrr(X, A, lamb, reg=0, alm_type=0, display=False):
    Q = orth(A.T)
    B = A.dot(Q)

    if reg == 0:
        if alm_type == 0:
            Z, E = exact_alm_lrr_l21v2(X, B, lamb, display=display)
        else:
            Z, E = inexact_alm_lrr_l21(X, B, lamb, display=display)
    else:
        if alm_type == 0:
            Z, E = exact_alm_lrr_l1v2(X, B, lamb, display=display)
        else:
            Z, E = inexact_alm_lrr_l1(X, B, lamb, display)

    Z = Q.dot(Z)
    return (Z, E)

