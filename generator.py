import numpy as np
import scipy.sparse as sp

def test(x):
    return x + 1

def east_sparse_generator(N, T):
    """
    Build sparse generator for East model (size 2^N).
    States are ints 0..2^N-1; bit i is site i (0=down,1=up)
    Rates: 1->0 : 1.0, 0->1 : exp(-1/T)
    Facilitation: site i flips only if site i-1 is 1 (with periodic boundaries)
    """
    k_down = 1.0
    k_up  = np.exp(-1.0/T)
    S = 1 << N
    rows, cols, vals = [], [], []

    def bit(b, i): return (b >> i) & 1

    for state in range(S):
        
        row_sum = 0.0
        
        for i in range(N):
            
            allowed = bit(state, (i-1) % N)
            if not allowed:
                continue
                
            spin = bit(state, i)
            target = state ^ (1 << i)
            rate = k_down if spin == 1 else k_up
            
            rows.append(state)
            cols.append(target)
            vals.append(rate)

            row_sum += rate

        rows.append(state)
        cols.append(state)
        vals.append(-row_sum)

    Q = sp.csr_matrix((vals, (rows, cols)), shape=(S, S), dtype=float)
    return Q