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

    return numerator / (Z-1)

def generate_configuration(rng, N, T):
    """
    Generates a single East model configuration of N lattice sites at
    temperature T.

    Returns: (N,) boolean jax array with True at excited sites.
    """
    p = concentration(T)
    lattice = jax.random.bernoulli(rng, p, (N,), mode='high').astype(bool)
    return lattice

def generate_walkers(rng, num_walkers, N, T):
    """
    Generates `num_walkers` East model configurations of N lattice sites at
    temperature T.

    Returns: (num_walkers,N) boolean jax array with True at excited sites.
    """
    rngs = jax.random.split(rng, num=num_walkers)
    func = jax.vmap(generate_configuration, in_axes=(0,None,None))
    return func(rngs, N, T)

def gillespie_dynamics_configuration(rng, configuration, T, num_steps):
    """
    Runs Gillespie dynamics for the east model for num_steps steps at
    temperature T, starting from the specified configuration.

    Returns:
      wait_times : (num_steps,) array of wait times in each configuration
      states :     (num_steps,N) array of configurations
    """

    k_down = 1.0
    k_up = jnp.exp(-1/T)
    N = configuration.shape[0]

    def compute_site_rates(state):
        allowed = jnp.roll(state, 1)
        rates = jnp.where(state, k_down, k_up) * allowed
        return rates

    def gillespie_step(carry, _):

        rng, state = carry
        
        new_rng, rng1, rng2 = jax.random.split(rng, 3)

        site_rates = compute_site_rates(state)
        total_rate = jnp.sum(site_rates)

        u = jax.random.uniform(rng1)
        dt = -jnp.log(u) / total_rate

        probs = site_rates / total_rate
        chosen = jax.random.choice(rng2, N, p=probs)
        new_state = state.at[chosen].set(~state[chosen])

        return (new_rng, new_state), (dt, state)

    carry0 = (rng, configuration)
    _, (wait_times, states) = jax.lax.scan(
        gillespie_step, carry0, None, num_steps
    )
    
    return ( wait_times , states )

def gillespie_dynamics_walkers(rng, walkers, T, num_steps):
    """
    Runs Gillespie dynamics for the east model for num_steps steps at
    temperature T, starting from the specified walker configurations.

    Returns:
      wait_times : (num_walkers,num_steps) array of wait times in each
                    configuration
      states :     (num_walkers,num_steps,N) array of configurations
    """
    num_walkers = walkers.shape[0]
    rngs = jax.random.split(rng, num=num_walkers)
    func = jax.vmap(gillespie_dynamics_configuration, in_axes=(0,0,None,None))
    return func(rngs, walkers, T, num_steps)

def acf_configuration(times, wait_times, states):
    """
    Computes the correlation function for the given trajectory (specified by
    wait_times and states) at the specified times.

    wait_times has shape (num_steps,) and states has shape (num_steps,N)

    Returns: (ts,) float array where (ts,) is also the shape of times
    """
    
    (num_steps,N) = states.shape
    
    cum_times = jnp.cumsum(wait_times)

    flucs = states - jnp.mean(states)
    fluc_0 = flucs[0,:]
    C0 = jnp.mean(fluc_0 * fluc_0)

    def compute_corr_at_time(t):
        idx = jnp.searchsorted(cum_times, t, side='right')
        idx = jnp.clip(idx, 0, num_steps-1)
        fluc_t = flucs[idx]
        return jnp.mean(fluc_0 * fluc_t) / C0

    return jax.vmap(compute_corr_at_time)(times)

def acf_walkers(times, wait_times, walkers):
    """
    Computes the correlation function for the given trajectories (specified by
    wait_times and states) at the specified times.

    wait_times has shape (num_walkers,num_steps) and states has shape
    (num_walkers,num_steps,N)

    Returns: (ts,) float array where (ts,) is the shape of times
    """
    func = jax.vmap(acf_configuration, in_axes=(None,0,0))
    acfs = func(times, wait_times, walkers)
    return jnp.average(acfs, axis=0)

def site_probabilities_walkers(times, wait_times, walkers):
    """
    Computes the probabilities for each site to be excited at the specified
    times.

    wait_times has shape (num_walkers,num_steps) and states has shape
    (num_walkers,num_steps,N)

    Returns: (ts,N) float array where (ts,) is the shape of times
    """

    num_walkers, num_steps, N = walkers.shape

    cum_times = jnp.cumsum(wait_times, axis=1)

    def site_probs_at_single_time(t):
        def single_walker_probs(cum_time_w, walker_w):
            idx = jnp.searchsorted(cum_time_w, t, side='right')
            idx = jnp.clip(idx, 0, num_steps-1)
            return walker_w[idx]
        states_at_t = jax.vmap(single_walker_probs)(cum_times, walkers)
        return jnp.mean(states_at_t, axis=0)

    return jax.vmap(site_probs_at_single_time)(times)

