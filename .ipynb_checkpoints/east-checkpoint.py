import jax
import jax.numpy as jnp

def concentration(T):
    """
    Computes the concentration of excitations in an East model at the specified
    temperature T.
    """
    return 1 / (1 + jnp.exp(1/T))

def finite_size_concentration(N,T):
    """
    Equilibrium <c> for N spins at temperature T, EXCLUDING the all-zero state.
    Energy E = number of up spins (non-interacting).
    Returns fraction of up spins.
    """
    infinite_concentration = concentration(T)
    Z = (1 + jnp.exp(-1/T)) ** N
    numerator = infinite_concentration * Z

    if jnp.isinf(numerator):
        return infinite_concentration
    else:
        return numerator / (Z-1)