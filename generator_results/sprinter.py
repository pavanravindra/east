import sys, os

sys.path.append("/burg-archive/ccce/users/phr2114/east")

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigs

import generator

N = 12
num_Ts = 16

Ts = 1.0 / np.linspace(0.01, 6, num_Ts)
results = np.full((Ts.size,2), np.nan)

results[:,0] = Ts

for (i,T) in enumerate(Ts):

    Q = generator.east_sparse_generator(N, T)
    vals, vecs = eigs(Q, k=16, which='LR')
    vals = np.sort(np.real(vals))
    slowest_mode = np.max(vals[np.where(vals < -1e-14)[0]])

    results[i,1] = -1.0 / slowest_mode

    print("{:.3f}\t{:.3e}".format(results[i,0], results[i,1]))

np.savetxt("timescales.txt", results)
