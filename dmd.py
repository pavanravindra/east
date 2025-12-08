import jax
import jax.numpy as jnp

def ExactDMD(X, Y, r=None):
    """
    Computes exact DMD eigenvalues and modes.
    
    X and Y should have shape (d,M) where d is the dimensionality of the data
    and M is the number of snapshot pairs.
    """

    if r is None:
        r = X.shape[0]

    U, S, Vh = jnp.linalg.svd(X)
    
    U = U[:,:r]
    S_mat = jnp.diag(S[:r])
    S_inv = jnp.diag(1 / S[:r])
    V = Vh[:r,:].T

    K_DMD = U.T @ Y @ V @ S_inv
    (evals, evecs) = jnp.linalg.eig(K_DMD)
    modes = Y @ V @ S_inv @ evecs
    
    return (evals,modes)
    