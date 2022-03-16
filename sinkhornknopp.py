import numpy as np
from time import time


def optimize_l_sk(prob, lmd, ddtype=np.float64):
    tt = time()

    n = prob.shape[0]
    k = prob.shape[1]

    prob = ddtype(prob)
    prob = prob.T  # (k, n)
    r = np.ones((k, 1), dtype=ddtype) / k
    c = np.ones((n, 1), dtype=ddtype) / n
    prob **= lmd  # (k, n)
    inv_k = ddtype(1. / k)
    inv_n = ddtype(1. / n)
    err = 1e6
    cnt = 0

    while err > 1e-1:
        r = inv_k / (prob @ c)  # (k, n) @ (n, 1) = (k, 1)
        c_new = inv_n / (r.T @ prob).T  # ((1, k) @ (k, n)).t() = (n, 1)
        if cnt % 10 == 0:
            err = np.nansum(np.abs(c / c_new - 1))
        c = c_new
        cnt += 1
    print("sinkhornknopp: error: ", err, 'step ', cnt, flush=True)  # " nonneg: ", sum(I), flush=True)

    # inplace calculations
    # ---
    prob *= np.squeeze(c)
    prob = prob.T
    prob *= np.squeeze(r)  # (n, k)
    # prob = prob.T  # (k, n)
    argmaxes = np.nanargmax(prob, axis=1)  # (n,)

    print('opt took {0:.2f}min, {1:4d}iters'.format(((time() - tt) / 60.), cnt), flush=True)

    return prob, argmaxes
