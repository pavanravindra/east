import jax
import jax.numpy as jnp
from functools import partial


# ---------- Thermodynamics ----------

def concentration(T):
    """Equilibrium concentration of excitations: c = 1 / (1 + e^{1/T})."""
    return 1.0 / (1.0 + jnp.exp(1.0 / T))


def temperature_from_length(l):
    """Temperature whose equilibrium concentration is c = 1/l."""
    return 1.0 / jnp.log(l - 1.0)


def finite_size_concentration(N, T):
    """
    Equilibrium <c> for N non-interacting spins at temperature T,
    conditioned on the state not being all zeros (absorbing state excluded).
    Computed in log-space to stay numerically stable at low T.
    """
    c = concentration(T)
    log_p_zero = N * jnp.log1p(-c)
    log_Z      = N * jnp.log1p(jnp.exp(-1.0 / T))
    return c / (1.0 - jnp.exp(log_p_zero - log_Z))



# ---------- Initial conditions ----------

def sample_equilibrium(N, T, key):
    """Sample an independent-spin equilibrium configuration of length N."""
    c = concentration(T)
    return jax.random.bernoulli(key, p=c, shape=(N,)).astype(jnp.int32)


# ---------- Gillespie dynamics ----------
#
# Standard MC wastes O(exp(1/T)) attempts per accepted flip at low T.
# Gillespie skips directly to the next event: much more efficient at low T
# and gives a physically meaningful continuous-time axis.
#
# Rates (Metropolis, satisfies detailed balance):
#   facilitated s_i = 0  →  1  at rate  exp(-1/T)   (= c/(1-c))
#   facilitated s_i = 1  →  0  at rate  1
#   not facilitated       →      rate  0
#
# Facilitation: spin i requires RIGHT neighbor s[i+1] = 1.
# Periodic boundary conditions: the right neighbor of spin N-1 is spin 0.

def gillespie_step(state, T, key):
    """
    One Gillespie event: draw the waiting time, choose the flipping site,
    and update the configuration.

    Returns
    -------
    new_state : updated spin array (same dtype/shape as state)
    dt        : physical time elapsed (exponentially distributed, mean 1/R)
    """
    # Right-neighbor facilitation mask
    right      = jnp.roll(state, -1)
    facilitated = (right == 1).astype(jnp.float32)

    rates = jnp.where(state == 0, jnp.exp(-1.0 / T), 1.0) * facilitated
    R     = jnp.sum(rates)

    k_time, k_site = jax.random.split(key)
    dt = -jnp.log(jax.random.uniform(k_time)) / R

    # Select site by inverse CDF (O(N), avoids jax.random.choice version issues)
    u    = jax.random.uniform(k_site)
    site = jnp.argmax(jnp.cumsum(rates) / R > u)

    return state.at[site].set(1 - state[site]), dt


# ---------- Trajectory generation ----------

@partial(jax.jit, static_argnums=(2, 3))
def _gillespie_trajectory(T, state_init, dt_record, n_records, key):
    """
    JIT core: advance the Gillespie process to fixed physical checkpoints
      t = dt_record, 2*dt_record, ..., n_records * dt_record
    and record the state at each one.

    Outer jax.lax.scan loops over checkpoints; inner jax.lax.while_loop
    fires Gillespie events until the next checkpoint is reached.

    static_argnums: dt_record (2) and n_records (3) so JAX can fix the
    scan length and treat dt_record as a compile-time constant.

    Returns
    -------
    traj : (n_records, N) int32  -- state at each checkpoint
    """
    def advance_to(carry, t_target):
        state, t, key = carry

        def cond(inner):
            _, t_inner, _ = inner
            return t_inner < t_target

        def body(inner):
            state, t_inner, key = inner
            key, subkey = jax.random.split(key)
            state, dt   = gillespie_step(state, T, subkey)
            return state, t_inner + dt, key

        state, t_new, key = jax.lax.while_loop(
            cond, body, (state, t, key)
        )
        return (state, t_new, key), state

    t_targets  = jnp.arange(1, n_records + 1, dtype=jnp.float32) * dt_record
    init_carry = (state_init, jnp.float32(0.0), key)
    _, traj    = jax.lax.scan(advance_to, init_carry, t_targets)
    return traj  # (n_records, N)


def generate_trajectory(N, T, dt_record, n_records, key, state_init=None):
    """
    Generate a Gillespie trajectory from a single initial configuration,
    prepending the t=0 state so that traj[0] is always the exact start.

    Parameters
    ----------
    N, T        : system size and temperature
    dt_record   : physical time between recorded snapshots
    n_records   : number of snapshots after t=0
    key         : JAX PRNG key
    state_init  : (N,) int32 array, or None to sample from equilibrium

    Returns
    -------
    traj  : (n_records + 1, N) int32
            traj[k] is the state at time k * dt_record
    """
    init_key, run_key = jax.random.split(key)
    if state_init is None:
        state_init = sample_equilibrium(N, T, init_key)
    traj = _gillespie_trajectory(T, state_init, dt_record, n_records, run_key)
    return jnp.concatenate([state_init[None], traj], axis=0)  # (n_records+1, N)


# ---------- Isoconfigurational ensemble ----------

def isoconfigurational_ensemble(N, T, dt_record, n_records, n_walkers,
                                 state_init, key):
    """
    Run n_walkers independent Gillespie trajectories from the SAME initial
    configuration state_init (the isoconfigurational ensemble).

    Returns
    -------
    mean_traj : (n_records + 1, N)
                propensity <s_i(t)> averaged over walkers;
                decays from state_init toward [c, c, ..., c]
    all_trajs : (n_walkers, n_records + 1, N)
                individual walker trajectories
    """
    keys = jax.random.split(key, n_walkers)

    # Each walker gets a different key but the same state_init
    def one_walker(k):
        traj = _gillespie_trajectory(T, state_init, dt_record, n_records, k)
        return jnp.concatenate([state_init[None], traj], axis=0)

    all_trajs = jax.vmap(one_walker)(keys)          # (n_walkers, n_records+1, N)
    mean_traj = jnp.mean(all_trajs, axis=0)          # (n_records+1, N)
    return mean_traj, all_trajs


# ---------- Autocorrelation ----------

def site_autocorrelation(traj, c_eq):
    """
    Normalized, site-averaged autocorrelation from a single trajectory.

        C(t) = [<s_i(0) s_i(t)>_sites - c^2] / (c - c^2)

    traj  : (n_records + 1, N)   (traj[0] must be the t=0 state)
    Returns C : (n_records + 1,),  C[0] = 1 by construction.
    """
    s0  = traj[0]
    raw = jnp.mean(s0[None, :] * traj, axis=1)
    return (raw - c_eq ** 2) / (c_eq - c_eq ** 2)


def ensemble_autocorrelation(N, T, dt_record, n_records, n_runs, key):
    """
    C(t) and relaxation-time statistics over n_runs independent equilibrium
    trajectories. Trajectories are generated once.

    tau is extracted from the *mean* C(t) so that the rescaled-time ACF plot
    passes through (t/tau=1, C=1/e) by construction. tau_err is the std of
    per-run crossing times divided by sqrt(n_runs), giving a meaningful error
    bar even though the point estimate uses the mean curve.

    Returns
    -------
    times   : (n_records + 1,)
    C       : (n_records + 1,)  mean autocorrelation
    C_err   : (n_records + 1,)  standard error over runs
    tau     : scalar            crossing of mean C at 1/e (nan if not reached)
    tau_err : scalar            std of per-run crossings / sqrt(n_valid)
    """
    c_eq     = concentration(T)
    run_keys = jax.random.split(key, n_runs)

    trajs = jax.vmap(
        lambda k: generate_trajectory(N, T, dt_record, n_records, k)
    )(run_keys)                                          # (n_runs, n_records+1, N)

    Cs     = jax.vmap(lambda t: site_autocorrelation(t, c_eq))(trajs)
    times  = jnp.arange(n_records + 1, dtype=jnp.float32) * dt_record
    C_mean = jnp.mean(Cs, axis=0)
    C_err  = jnp.std(Cs, axis=0) / jnp.sqrt(n_runs)

    tau     = relaxation_time(times, C_mean)
    taus    = jax.vmap(lambda C: relaxation_time(times, C))(Cs)
    n_valid = jnp.sum(~jnp.isnan(taus)).astype(jnp.float32)
    tau_err = jnp.where(n_valid > 1, jnp.nanstd(taus) / jnp.sqrt(n_valid), jnp.nan)

    return times, C_mean, C_err, tau, tau_err


# ---------- Relaxation time ----------

def relaxation_time(times, C, level=1.0 / jnp.e):
    """
    Extract tau by linear interpolation at the first crossing of C = level.
    Returns tau in physical Gillespie time units, or jnp.nan if not reached.
    """
    below   = C < level
    idx     = jnp.argmax(below)
    crossed = jnp.any(below)
    t1, t2  = times[idx - 1], times[idx]
    C1, C2  = C[idx - 1], C[idx]
    tau     = t1 + (level - C1) * (t2 - t1) / (C2 - C1)
    return jnp.where(crossed, tau, jnp.nan)
