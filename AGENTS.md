# AGENTS.md

## Project Overview

This project applies Dynamic Mode Decomposition (DMD) to the 1D East model — a kinetically constrained spin model exhibiting fragile glassy dynamics — to extract relaxation timescales, characterize the spectrum of decay modes, and test generalization (across initial conditions and forward in time). The methodology closely follows Nicolaou, Cho, Zhang, Kutz & Brunton (2026), *Signature of Glassy Dynamics in Dynamic Mode Decompositions* (arXiv:2502.10918v3), adapted from deterministic oscillator systems to a stochastic Markov system.

**Reference implementation for DMD machinery:** https://github.com/znicolaou/kuramoto_dmd

**Physical model reference:** Buhot & Garrahan (2002), *Crossover from fragile to strong glassy behaviour in the spin facilitated chain model* (J. Phys.: Condens. Matter 14, 1499).

---

## Repository Layout

- **`east.py`** — core East model code: continuous-time simulation, the kinetic constraint, and the isoconfigurational (W-walker) ensemble machinery.
- **`ACF.ipynb`** — numerical calculation of the equilibrium autocorrelation function and extraction of tau_ACF.
- **`Isoconfigurational.ipynb`** — isoconfigurational-ensemble propensity trajectories and tau_iso extraction.
- **`DMD.ipynb`** — baseline DMD calculations (single-site dictionary, single temperature).
- **`DMD_n_walkers.ipynb`** — walker-count study; demonstrates eigenvalue convergence onto the real axis as W increases.
- **`DMD_temperatures.ipynb`** — temperature-dependent spectra (the l = 3, 4, 6, 8 series).

All current DMD notebooks use the **single-site (L = 1) dictionary**. This is the *baseline*, not the finished method — see "Dictionary Construction" and "Next Steps." A future agent should not mistake the single-site spectra for the final result.

---

## Physical System

**Hamiltonian:** H = sum_i s_i with s_i in {0,1} and field h = -1. Spins are non-interacting energetically, so the equilibrium distribution is a **product measure**.

**Equilibrium defect density:** c_eq = 1 / (1 + e^(1/T)).

**Dynamics (East model, fully asymmetric b=0):** continuous-time Monte Carlo. Spin s_i flips with rate

    P(s_i flip) = P_metro(dE) * s_{i+1},     dE = 1 - 2 s_i

so the spin can only flip if its right neighbor is a defect (kinetic constraint). The two microscopic rates are 1 (annihilation, 1->0, downhill) and e^(-1/T) (creation, 0->1, uphill), each gated by the right-neighbor defect.

**Relevant scales:**
- Equilibrium length: l_eq = 1/c_eq ~ e^(1/T)
- Barrier hierarchy: tau_k ~ e^(k/T) for relaxation across chains of length 2^(k-1) <= l < 2^k
- Asymptotic relaxation time: tau_AS ~ exp(A/T^2) with A = 1/ln 2 (Bassler super-Arrhenius)

**Why DMD applies despite stochastic dynamics:** the stochastic Koopman operator (K^t g)(s0) = E[g(s(t)) | s(0) = s0] is linear. The isoconfigurational ensemble — many walker trajectories from the same initial configuration with different PRNG seeds — is a Monte Carlo estimate of this operator's action on observables. DMD applied to walker-averaged dictionary trajectories approximates the stochastic Koopman operator.

**The spectrum is effectively real (established result).** The East model is a reversible Markov chain (detailed balance with respect to its product-measure equilibrium), so its generator is self-adjoint in L2(pi) and its true Koopman spectrum is purely real — there are no genuine oscillatory modes. This is confirmed empirically in `DMD_n_walkers.ipynb`: as W increases, the imaginary parts of the DMD eigenvalues collapse onto the real axis. Off-axis structure at low W is a finite-sampling artifact of the empirical inner product, not physics, and it shrinks with better expectation estimates. A real finite K still produces conjugate pairs, but for this system those pairs should be read as the finite approximation's attempt to resolve nearby *real* eigenvalues, and their imaginary parts vanish under refinement.

---

## Data Generation Pipeline

For each target temperature T:

1. **Sample n_configs initial configurations from equilibrium at T.** Default: n_configs = 16. Use a long equilibrium pre-thermalization, then take independent snapshots. Chain length N must satisfy N >> l_eq(T); recommend N >= 50 * l_eq, capped by compute budget. All trajectories at a given T must share the same N for pooling.

2. **For each initial configuration, run W walkers** with distinct PRNG seeds, all starting from the same configuration. Walkers must share the initial state exactly and differ only in their stochastic realization. W controls the variance of the expectation estimate; see "Implementation Notes" for the convergence diagnostic.

3. **Sample on a uniform time grid** with timestep dt. DMD assumes fixed-dt snapshot pairs.
   - T_window must be long enough to see the slowest mode of interest. The resolution floor 1/T_window is fundamental (see "Fundamental Limits") and becomes the binding constraint at low T.
   - dt should resolve the fastest dynamics (microscopic timescale ~1).
   - At low T, dynamic range is enormous; prioritize sufficient T_window over fine dt.

4. **Output layout:** (n_configs, W, T_times, N) array of binary spin values.

5. **Sampling regime:** equilibrium-sampled initial configs run at the same T (NOT a quench). The product-measure equilibrium and the kinetic-constraint dynamics jointly satisfy detailed balance.

**Held-out configurations.** Generating fresh equilibrium configurations is trivial for the East model. Reserve a held-out set (new configs, each with its own W walkers) that are *not* used in the DMD fit. These are used for out-of-sample reconstruction tests — see "Next Steps" step 2 and Goal 4. Held-out evaluation requires walker-averaged trajectories (we compare to E[g | s(0)], not a single noisy realization), so new configs must also be run with W walkers.

---

## Dictionary Construction

The dictionary observables are the binary analog of the Brunton paper's Fourier basis: Walsh-Hadamard monomials in the binary variables.

**Structure (downward-closed, local contiguous clusters up to size L):**

| Order | Form                              | Count   |
|-------|-----------------------------------|---------|
| 0     | {1}                               | 1       |
| 1     | {s_i}                             | N       |
| 2     | {s_i s_{i+1}}                     | N-1     |
| 3     | {s_i s_{i+1} s_{i+2}}             | N-2     |
| ...   | ...                               | ...     |
| L     | {s_i s_{i+1} ... s_{i+L-1}}       | N-L+1   |

Total dictionary size: D = 1 + sum_{k=1}^L (N - k + 1).

**Important constraints:**
- **Downward closure** (include all sub-clusters) is required so that {0,1} and {-1,+1} encodings span the same function space (they differ by an invertible affine transformation requiring lower-order terms to be present).
- **Locality:** use contiguous clusters only. The East model's facilitation is local and directional; relevant correlations are local. Do NOT use random pairs (that would mirror Daido's all-to-all coupling structure, which is wrong for this system).
- **L is a free parameter** to scan.

**Why higher L is expected to matter (and what "improvement" looks like).** The single-site dictionary is *not closed under the generator*. Working out the generator on a single occupation gives

    d/dt E[s_j] = e^(-1/T) E[s_{j+1}] - (1 + e^(-1/T)) E[s_j s_{j+1}]

so the evolution of a single-site quantity depends on the pair quantity E[s_j s_{j+1}], which is outside the L = 1 span; the next order couples to triples, and so on (a BBGKY-style hierarchy). A truncated dictionary cuts this hierarchy and projects the leftover onto what it has. The concrete, falsifiable prediction is that at low T (large l) the single-site dictionary will **under-resolve the slow cooperative modes**, and adding L = 2, L = 3 should (i) push resolved modes closer to Re(mu) = 0 / extend the slow tail, (ii) lower the residuals eps of the slow modes, and (iii) bring the reconstructed-trajectory decay (see Next Steps step 2) into closer agreement with tau_ACF. This both improves the method and is the test for Goal 2 (dictionary length scale growing with l_eq).

---

## Critical Implementation Detail: Average Products Over Walkers

The single most common implementation error to avoid.

The Koopman operator evolves expectations:

    E[g(s(t+dt)) | s(0)] ~ K * E[g(s(t)) | s(0)]

For product observables this means

    E[s_i(t) s_j(t) | s(0)]   !=   E[s_i(t) | s(0)] * E[s_j(t) | s(0)]

The difference is the connected correlation, which carries the cooperative physics.

**Correct pipeline:**
1. Evaluate the dictionary on each walker's binary configuration at each time -> (n_configs, W, T_times, D).
2. Average over the W axis -> (n_configs, T_times, D). These are estimates of the expected dictionary vector.

**Wrong pipeline:** computing single-site propensities p_i(t) = (1/W) sum_w s_i^w(t) first and then forming products of propensities. This omits the connected correlation entirely.

---

## Centering and Scaling

Center and scale each dictionary observable by its **equilibrium** mean and standard deviation — never the empirical window mean.

**Why equilibrium, not window:** subtracting the window mean of a partially-relaxed observable centers it mid-decay, which causes DMD to manufacture spurious oscillatory modes (a known DMD pathology with temporally mean-centered data). Subtracting a constant (the equilibrium value) shifts where the constant-mode amplitude sits and leaves all decaying modes unchanged.

**Computing equilibrium statistics exactly (analytic, no estimation noise):** the East model equilibrium is a product measure, so

    <prod_{i in S} s_i>_eq = c_eq^|S|

and equilibrium variances are products of independent-Bernoulli formulas:

    Var(s_i)_eq = c_eq (1 - c_eq)
    Var(prod_{i in S} s_i)_eq = c_eq^|S| (1 - c_eq^|S|)

For each dictionary observable g_k with equilibrium mean mu_k and equilibrium std sigma_k:

    g_k^standardized(s) = (g_k(s) - mu_k) / sigma_k

This transformation is a constant per observable, commutes with walker-averaging, and can be applied at any stage of the pipeline.

**Keep the constant observable {1} in the dictionary even after centering.** It is the trivially conserved Koopman eigenfunction (mu = 0) — necessary for operator closure and conditioning. After centering, its amplitude on each non-constant observable's reconstruction should be near zero, which is correct.

---

## DMD Pipeline (mirror the Brunton paper)

1. **Pool snapshot pairs across all initial configurations.** Stack the (T_times - 1) pairs from each of n_configs trajectories into joint Psi_X, Psi_Y matrices of shape (D, n_configs * (T_times - 1)). Always pool; do not fit per-trajectory (see Implementation Notes).

2. **Exact DMD with SVD truncation.** Truncate to rank r retaining a target fraction of singular-value energy (start with ~99%). Form the reduced operator A_tilde = U* Y V Sigma^-1 and compute its eigendecomposition.

3. **Continuous-time eigenvalues:** mu_i = log(lambda_i) / dt.

4. **resDMD residual filter.** For each eigenpair, compute the residual eps_i. Keep modes with eps_i < eps_threshold. Start with eps_threshold = 5e-8 (Brunton's value); tune.

5. **Pseudospectra.** Compute eps-pseudospectrum on a grid in the complex mu-plane for visualization (a la Brunton Fig. 4).

6. **Mode amplitudes per trajectory:** project initial state onto eigenvectors to get b_i^j(0). Reconstruction:

       g_hat_k^j(t) ~ sum_i Phi_tilde_k^i b_i^j(0) e^(mu_i t)

7. **Relevance filter (optional second stage):** among low-eps modes, keep the minimum number required to reconstruct trajectories to a target relative accuracy (start with 1e-3, Brunton's value). Measure mode contribution as **integrated effect over the window**, not raw |b_i(0)|. **Caution:** do not threshold away slow modes — they carry the scientifically important long-time behavior. Apply the relevance filter mainly to prune the fast/noisy bulk.

For implementation details (exact-DMD with SVD projection, resDMD residual computation, pseudospectrum evaluation), follow the reference repository. Whether large-D cases require chunked linear algebra (dask) is not yet clear; extend to dask only if memory/runtime demands it.

---

## What the Spectrum Looks Like (preliminary, single-site)

From the single-site temperature series (`DMD_temperatures.ipynb`, l = 3, 4, 6, 8):

**High T (e.g. l = 3, T ~ 1.443):** a clear two-domain structure. A fast cluster sits well to the left at the bare facilitated rate; a cooperative cluster sits between roughly -e^(-1/T) and 0, with the cooperative tiers (tau_k ~ e^(k/T) -> rates e^(-k/T)) partially resolvable as distinct sub-clusters, and a gap between the fast and cooperative domains.

**Decreasing T (l = 4 -> 6 -> 8):** the resolved modes move toward Re(mu) = 0 (everything slows), and the distinct cooperative tiers merge into a continuum that hugs the imaginary axis. This accumulation toward Re(mu) = 0 with no clean gap is the emerging glassy / stretched-exponential signature, directly analogous to Brunton's "no-gap" algebraic case. The successive tiers e^(-k/T) compress geometrically toward zero (each ~e^(-1/T) times the previous), so at low T they crush together near the axis.

**The e^(-1/T) reference line.** e^(-1/T) marks the **leading activated rate** — the fastest of the cooperative modes (it equals 1/tau_1, the k = 1 rate). Treat it as the fast cutoff for where slow, activated dynamics begin: anything faster than e^(-1/T) is constraint-free microscopic relaxation that is not relevant to the glassy physics, and the activated band begins at e^(-1/T) and runs toward zero. At high T it visually separates the two domains; at low T it stops being a clean separator and becomes the *upper edge* of the cooperative continuum (the k >= 2 modes fill in below it). **Open question:** the precise status of this line and whether it is the right principled cutoff is not settled. It is recorded here because it is a useful reference and a candidate filter for excluding fast, non-glassy modes — keep it in mind but do not over-rely on it.

**Caveats on the low-T panels.** The accumulation near zero at l = 6, 8 is exactly where two effects bite and must be ruled out before the near-zero density is read as physics: (i) the resolution floor 1/T_window — the slowest tiers may simply lie beyond the observation window; and (ii) single-site under-expressiveness — the cooperative modes at low T involve correlations over the growing length l_eq, which L = 1 cannot represent. Before trusting low-T conclusions, confirm the target timescales lie inside T_window, and check whether higher L sharpens/extends the slow tail.

---

## Next Steps

These three steps form a dependency chain and share a single concrete observable (the reconstructed-trajectory 1/e decay). Intended order: histogram modes by tier -> use the tier weighting to define a comparable timescale -> improve the slow tail with higher L. Do not extract a final timescale from under-resolved single-site spectra and conclude prematurely.

### Step 1 — Histogram modes by e^(-j/T) activation tiers

Cheapest thing to try, so do it first. Classify each resolved DMD mode by the integer activation tier j ~ -T * ln|Re(mu_i)| and histogram. **Bin in the integer tier index j, not in |Re(mu)| directly** — the tiers compress geometrically toward zero, so a linear binning will not separate them, but in j they are evenly spaced.

**Weighting:** the histogram weight per mode should be the same mode-importance object used for the Step 2 weighted timescale (amplitude / integrated-effect / residual-aware weight) — compute it once and reuse. A raw eigenvalue count conflates "many modes in a tier" with "this tier matters dynamically."

See how well the tiers resolve at least down to l = 8. If it works, "number of populated tiers" or "amplitude-weighted mean tier index" is a candidate data-driven order parameter — the East-model analog of Brunton's eta — and ties directly to Goal 3 (more tiers populating as T drops = spectral broadening).

### Step 2 — Extract a relevant timescale from the DMD spectra

Target: a tau_DMD comparable to tau_ACF. Be explicit about *which* timescale, since the spectrum has multiple natural scales (microscopic, leading activated e^(-1/T), slowest resolved mode). Two definitions to implement and compare:

- **(a) Reconstructed-trajectory 1/e time.** Reconstruct the walker-averaged ACF (or propensity) from the DMD modes and read off where it crosses 1/e. This is the most apples-to-apples comparison with tau_ACF (which is itself a 1/e crossing).
- **(b) Weighted effective rate.** A weighted average over modes, e.g. tau_eff = sum|b_i| / sum|b_i||Re(mu_i)|. The functional form is flexible — weights may incorporate amplitude, residual, and integrated effect; explore options and report the choice.

**Two reconstruction modes (this is the key methodological axis):**
- **In-sample:** reconstruct the same initial configurations used to fit the operator. Tests fit only.
- **Out-of-sample (held-out):** generate fresh equilibrium configurations (with their own W walkers) not used in the fit, and test how well the learned operator predicts their dynamics. This is the stringent test — it distinguishes a learned *operator* from memorized trajectories — and should be treated as the default validation, not an afterthought. (Requires walker-averaged trajectories for the new configs.)

**Define a reconstruction-quality scalar** (e.g. relative L2 error between reconstructed and measured walker-averaged ACF over the window) so that comparisons across L and across in/out-of-sample are quantitative, not just visual. This scalar is what Step 3 optimizes.

### Step 3 — Higher-order couplings (L = 2, 3, ...)

With the reconstructed-decay target from Step 2 in hand, the payoff of higher L becomes a clean, falsifiable observable. Expected signature in the large-l (low-T) limit: single-site held-out reconstruction should decay **too fast** (overshoot the true ACF) because the dictionary is not closed under the generator; adding L = 2, 3 should pull the reconstructed-1/e-time toward tau_ACF, lower the slow-mode residuals, and extend the resolved slow tail toward Re(mu) = 0. Track the reconstruction-quality scalar vs L at fixed T, most visibly at l = 8.

**Disambiguate the two causes of a too-fast low-T reconstruction before chasing L:** dictionary under-expressiveness (fixable with higher L) and slowest modes lying beyond 1/T_window (fixable only with longer trajectories). They look similar in the reconstructed decay but have different fixes. Confirm tau_ACF lies inside the observation window before attributing a fast-decaying reconstruction to dictionary limits.

---

## Standing Goals

### Goal 1 — tau_DMD with super-Arrhenius scaling

Extract tau_DMD across T (via Step 2) and compare against:
- **tau_iso** — 1/e-crossing of the isoconfigurational propensity decay (`Isoconfigurational.ipynb`).
- **tau_ACF** — 1/e-crossing of the equilibrium spin autocorrelation (`ACF.ipynb`).

**Validation target:** all three should exhibit Bassler scaling log tau ~ A/T^2 with A ~ 1/ln 2 ~ 1.44. They will NOT necessarily agree in absolute value: tau_iso and tau_ACF are 1/e summaries of stretched-exponential decay, while a slowest-mode tau_DMD is the asymptotic tail timescale. The scaling collapse (parallel slopes on log tau vs 1/T^2) is the claim. The reconstructed-ACF 1/e definition (Step 2a) is the closest match in absolute value.

### Goal 2 — Dictionary length scale grows with l_eq

Via Step 3: scan L at each T and identify the minimum L at which the slow modes resolve / the reconstruction-quality scalar saturates. Hypothesis: L_min(T) ~ l_eq(T) ~ e^(1/T), analogous to the rank-K dependence in Brunton's Daido model. This is the cleanest structural result of the project.

### Goal 3 — Progressive spectral broadening with decreasing T

As T decreases, more tiers of the barrier hierarchy populate and the spectrum accumulates toward Re(mu) = 0 (already visible in the single-site series). Quantify via the Step 1 tier histogram (number / weighted index of populated tiers) and the density of modes near the imaginary axis.

**Note:** the East model is stretched-exponential at all temperatures in the asymmetric limit (always fragile). What changes with T is the *width* of the spectrum and the number of populated barrier tiers, NOT a sharp exponential->stretched transition. Frame results accordingly.

### Goal 4 — Generalization: across configurations and forward in time

Two axes of the same question, organized as a 2x2 battery: **{train configs, held-out configs} x {full window, truncated window}**.

- *Held-out configs* (Step 2 out-of-sample) tests whether the operator generalizes across initial conditions.
- *Truncated window* (train on early-time data, predict the long-time tail) tests temporal extrapolation.
- The most stringent cell — **held-out configs + truncated window** — is the real prize: if the learned operator predicts the slow decay of a fresh configuration from only early-time training data, that is a strong result and the practical payoff (extrapolating tau to temperatures where direct simulation cannot reach full relaxation).

Extrapolation performance is measured on the **predicted reconstructed-trajectory averages** (the walker-averaged ACF / propensity), using the reconstruction-quality scalar from Step 2. The resolution floor is the central scientific limit here: see "Fundamental Limits."

---

## Fundamental Limits

- **Resolution floor:** modes with timescale exceeding T_window appear nearly constant over the window and cannot be reliably distinguished from the mu = 0 mode. The conservative systematic error bound is 1/T_window (Brunton's grey bands in their Fig. 5). This is the binding constraint at low T and must be checked before reading the near-zero accumulation as physics.

- **Spurious positive-Re(mu) eigenvalues:** some DMD eigenvalues will appear with Re(mu) > 0 even after residual filtering — forbidden growing modes that are finite-data artifacts of non-normality. Use the spread of their real parts as an empirical random-error estimate (Brunton's eta error-bar construction).

- **Conditioning at low T:** product observables have equilibrium means c_eq^|S|, which is tiny at low T. Equilibrium-mean centering and equilibrium-std scaling are essential for numerical stability; without them the SVD truncation behaves erratically.

- **Memory/runtime at large L and N:** dictionary size D = O(N * L). Use chunked linear algebra (dask) only if needed.

---

## Implementation Notes

- **Encoding ({0,1} vs {-1,+1}):** with a downward-closed dictionary these are affine-equivalent and produce identical spectra in exact arithmetic. Use {0,1} (simpler) and rely on equilibrium-centering/scaling for conditioning. Do NOT switch encodings expecting more signal — the affine equivalence means signal is relocated, not added.

- **Walker count W:** controls variance of dictionary averages. Established diagnostic (`DMD_n_walkers.ipynb`): as W increases, eigenvalue imaginary parts collapse onto the real axis and the resolved spectrum stabilizes. Increase W until the slow modes (the ones that matter) are stable.

- **n_configs:** few initial conditions are needed because each large-N chain contains many statistically-independent local relaxation environments (spatial extent substitutes for an ensemble of ICs, via approximate translation invariance). Brunton used 5 ICs in the minimal example and 3 in the oscillator glass. 16 is generous; do not over-invest in this knob. (Note: held-out evaluation needs *additional* fresh configs beyond the fitting set.)

- **Translation invariance:** the dynamics is statistically translation-invariant, so the operator K has approximate translational structure. We do NOT enforce this; we let DMD recover it. Near-degenerate clusters of eigenvalues (translates of the same local mode) are expected and benign.

- **Pooling vs per-trajectory DMD:** always pool snapshot pairs from all ICs into one fit per temperature. Per-trajectory DMD on single isoconfigurational trajectories is poorly conditioned (a single relaxation traces out a near-low-dimensional curve in dictionary space and fails to excite the full mode structure).

---

## What This Project Is Not

- It is not a re-derivation of DMD theory. Use the Brunton paper and reference implementation; deviate only where the binary/stochastic setting requires it.
- It is not a study of the fragile-to-strong crossover. That requires the intermediate-b model (Buhot-Garrahan section 3.3), a possible future extension. The pure East model (b = 0) is fragile at all T.
- It is not a quench study. All data is equilibrium-sampled and run at the same T. Quench dynamics violate the equilibrium length-distribution stationarity that underpins the relaxation timescale predictions.