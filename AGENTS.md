# AGENTS.md

## Project Overview

This project applies Dynamic Mode Decomposition (DMD) to the 1D East model — a kinetically constrained spin model exhibiting fragile glassy dynamics — to extract relaxation timescales, characterize the spectrum of decay modes, and recover stretched-exponential behavior. The methodology closely follows Nicolaou, Cho, Zhang, Kutz & Brunton (2026), *Signature of Glassy Dynamics in Dynamic Mode Decompositions* (arXiv:2502.10918v3), adapted from deterministic oscillator systems to a stochastic Markov system.

**Stay close to the Brunton-group method.** This is the only paper in the field on DMD for glassy systems, and its authors are authoritative on the DMD side (less so on glassy physics). Emulate their approach — exact DMD with energy-based SVD truncation, resDMD residual filtering, pseudospectra, initial-amplitude least-squares reconstruction — and deviate only where the binary/stochastic/spatial setting genuinely requires it. Do not import more sophisticated DMD variants unless a concrete failure forces it.

**Reference implementation for DMD machinery:** https://github.com/znicolaou/kuramoto_dmd

**Physical model reference:** Buhot & Garrahan (2002), *Crossover from fragile to strong glassy behaviour in the spin facilitated chain model* (J. Phys.: Condens. Matter 14, 1499).

---

## The Central Change From the Previous Setup (read this first)

Earlier versions of this project represented the **full N-site chain** as a single state vector in R^(N+1) (the +1 is the constant observable), with the attractor at [c_eq, ..., c_eq], and pooled over a modest number of time-evolved trajectories. **That representation has been retired.** It had two fatal problems:

1. **It does not respect translational invariance.** Two configurations differing only by a lattice shift (with PBCs) are physically equivalent local relaxation problems, but DMD on full-chain state vectors treats them as entirely independent trajectories. The representation does not describe the underlying physical system; it describes a particular labeling of it.

2. **Adding data degraded the fit instead of improving it.** With the full-chain state, each additional initial configuration is a genuinely distinct direction in R^(N+1). The exact-DMD SVD truncation (e.g. to ~80 modes) must then spend its rank budget *spanning configuration space* rather than *resolving timescales*. Empirically, 16 configs reconstructed cleanly but 64 configs reconstructed much worse — the truncation was forced to trade representation against dynamics. This is backwards: more data should sharpen the dynamics, not dilute the mode budget.

**The new representation is per-site and pooled.** Instead of evolving the full-chain state, we represent the **decay of individual sites**, dressing each site with the time-dependent states of the nearby spins that govern its relaxation. Concretely, the object that used to be "one trajectory" (one initial config, one chain of N sites) now supplies **N local trajectories** — one per site — that are pooled together. Two shift-related configurations now contribute the *same* local dynamical information rather than competing directions, so translational invariance is recovered by construction, and the representational burden no longer grows with the number of distinct configurations. This is the direct analog of Nicolaou et al. pooling over trajectories x^j(t); here we pool over both sites *and* initial configurations.

A consequence: **far fewer initial configurations are now needed**, because each large-N chain already supplies N pooled local samples. Do not over-invest in n_configs.

---

## Repository Layout

- **`east.py`** — core East model code: continuous-time simulation, the kinetic constraint, and the isoconfigurational (W-walker) ensemble machinery.
- **`ACF.ipynb`** — numerical calculation of the equilibrium autocorrelation function and extraction of tau_ACF.
- **`Isoconfigurational.ipynb`** — isoconfigurational-ensemble propensity trajectories and tau_iso extraction.
- **`DMD.ipynb`** — baseline DMD calculations (single-site dictionary, single temperature).
- **`DMD_n_walkers.ipynb`** — walker-count study; demonstrates eigenvalue convergence onto the real axis as W increases.
- **`DMD_temperatures.ipynb`** — temperature-dependent spectra (the l = 3, 4, 6, 8 series).
- **`DMD_temperatures_cropped.ipynb`** — pooled per-site diagnostic; currently uses the single-site (J = 0) local dictionary as the baseline of the new representation.

The single-site local dictionary in the current notebooks is the **J = 0, order-1 baseline** of the new per-site representation, not the finished method — see "Local Dictionary Construction" and "Phase 0 (Gating)." A future agent should not mistake the single-site spectra for the final result.

---

## Physical System

**Hamiltonian:** H = sum_i s_i with s_i in {0,1} and field h = -1. Spins are non-interacting energetically, so the equilibrium distribution is a **product measure**.

**Equilibrium defect density:** c_eq = 1 / (1 + e^(1/T)).

**Dynamics (East model, fully asymmetric b=0):** continuous-time Monte Carlo. Spin s_i flips with rate

    P(s_i flip) = P_metro(dE) * s_{i+1},     dE = 1 - 2 s_i

so the spin can only flip if its right neighbor is a defect (kinetic constraint). The two microscopic rates are 1 (annihilation, 1->0, downhill) and e^(-1/T) (creation, 0->1, uphill), each gated by the right-neighbor defect. Facilitation is therefore **directional**: a site's dynamics is controlled by the neighbors on its facilitating (right) side.

**Relevant scales:**
- Equilibrium length: l = 1/c_eq ~ e^(1/T). (Throughout this document, `l` denotes this East-model length scale — the inverse concentration. It is NOT the DMD window size; that is `J`.)
- In the East model the static and dynamic length scales coincide, so `l` is also the relevant correlation length for the local representation.
- Barrier hierarchy: tau_k ~ e^(k/T) for relaxation across chains of length 2^(k-1) <= l < 2^k
- Asymptotic relaxation time: tau_AS ~ exp(A/T^2) with A = 1/ln 2 (Bassler super-Arrhenius)

**Why DMD applies despite stochastic dynamics:** the stochastic Koopman operator (K^t g)(s0) = E[g(s(t)) | s(0) = s0] is linear. The isoconfigurational ensemble — many walker trajectories from the same initial configuration with different PRNG seeds — is a Monte Carlo estimate of this operator's action on observables. DMD applied to walker-averaged dictionary trajectories approximates the stochastic Koopman operator.

**The spectrum is effectively real (established result).** The East model is a reversible Markov chain (detailed balance with respect to its product-measure equilibrium), so its generator is self-adjoint in L2(pi) and its true Koopman spectrum is purely real — there are no genuine oscillatory modes. This is confirmed empirically in `DMD_n_walkers.ipynb`: as W increases, the imaginary parts of the DMD eigenvalues collapse onto the real axis. Off-axis structure at low W is a finite-sampling artifact of the empirical inner product, not physics, and it shrinks with better expectation estimates. A real finite K still produces conjugate pairs, but for this system those pairs should be read as the finite approximation's attempt to resolve nearby *real* eigenvalues, and their imaginary parts vanish under refinement.

---

## State Representation (per-site, pooled)

The fundamental object is no longer the full-chain state but a **local state vector attached to each site i**, built from walker-averaged, analytically-standardized cluster observables drawn from a local window.

**The local window.** For each site i, define a one-sided window of reach J on the **facilitating (right) side**:

    window(i) = {i, i+1, i+2, ..., i+J}

The window is asymmetric and includes only the facilitating side. Rationale: the goal is to predict the dynamics of site i by including a description of exactly those sites whose states govern i's relaxation. In the fully asymmetric East model, facilitation comes from the right, so only the right side is dynamically relevant to i. (Including the non-facilitating side would describe sites that i *influences* rather than sites that influence i; this is deliberately excluded as the cleaner picture. Revisit only if there is a concrete reason.)

**J replaces the old "full chain" extent.** J is a hyperparameter — the spatial reach of the local description. In the J -> infinity limit the local state contains all information about a site's neighborhood; in practice we expect quality to saturate once J reaches the correlation length l(T) (see Phase 0).

**Pooling.** Each site i in each initial configuration supplies one local trajectory of its standardized cluster vector through time. We pool **over both sites and initial configurations** into a single snapshot-pair set per temperature. If there are T_traj initial configurations, N sites, and D_t time records, this yields on the order of T_traj * N * D_t local state samples and T_traj * N * (D_t - 1) snapshot pairs. (Sites near the right boundary whose windows run off the chain are handled by PBCs, consistent with the shift-invariance the representation is built to respect.)

This pooling is what recovers translational invariance and what makes additional data sharpen rather than dilute the dynamics.

---

## Local Dictionary Construction

The dictionary observables are the binary analog of the Brunton paper's Fourier basis. Where Brunton expands a phase theta on a circle in Fourier modes e^(ik theta), we expand a local spin configuration on the Boolean hypercube {0,1}^(J+1) in **Walsh-Hadamard monomials** — products of spins drawn from the window. These are the complete orthogonal basis for functions on the local window, exactly as Fourier modes are complete for periodic functions. A function of the local configuration is a linear combination of monomials chi_S = prod_{k in S} s_k for subsets S of the window.

Your earlier hand-written list [1, <s_i>, <s_i s_{i+1}>, ...] was already the bottom of this basis; the Walsh-Hadamard framing names the complete set and gives a principled truncation order.

**Three truncation knobs.** The dictionary is controlled by three independent parameters:

1. **J — spatial reach.** Which sites the window includes: {i, ..., i+J}. Bounds the support of any monomial.
2. **p — coupling order.** The maximum number of spins appearing in a single monomial (|S| <= p). This is the analog of Brunton's Fourier truncation |k| <= d: low order = low-order spin correlations. Truncating by p is truncating by correlation order.
3. **Contiguity mode** — which subsets S within the window are admitted (see below).

Dictionary size is set jointly by (J, p, contiguity mode). Crucially, **spectral richness — the number of resolvable timescales — is set by dictionary size, not by the dimension of the raw local state.** A small local state lifted to a large dictionary has a rich Koopman spectrum (exactly as Brunton's 1D theta with a large Fourier dictionary resolves hundreds of decaying modes). This is why the dictionary, not the bare window, is the knob that governs how many timescales we can extract — and why a low-dimensional local state is not a limitation as long as the dictionary is expressive.

**Contiguity modes (the third knob), in order of increasing size:**

- **anchored-contiguous:** only contiguous runs that contain i itself: {i}, {i,i+1}, {i,i+1,i+2}, ... Size ~ J+2. This is the original purely "decay of site i dressed by its right-neighbors" picture. Leanest; likely too small to express complex dynamics at small l (few modes survive SVD truncation).
- **floating-contiguous (DEFAULT):** any contiguous run within the window, anchored at i or not (e.g. {i+2, i+3} is admitted). Size grows ~ linearly in J. After pooling over sites, a floating chain in site i's window is the same physical object as an anchored chain in another site's window, so floating-contiguous gives each site a more complete local description without breaking translation invariance.
- **all-subsets:** every subset of the window, including gapped clusters like {i+2, i+4, i+7}. Size grows combinatorially (full Walsh completeness at p = J+1 gives 2^(J+1)). Includes gapped terms that have no *direct* kinetic pathway in the East constraint but can be *induced* by the equation-of-motion hierarchy (e.g. {i+2, i+4} slaved through i+3).

**Default and diagnostics:** run with **floating-contiguous** as the working basis. During Phase 0, run anchored-contiguous and all-subsets as one-off comparisons at one or two representative temperatures, to *measure* whether the contiguity restriction is justified rather than assuming it. Expectation (to be confirmed): gapped all-subsets terms move reconstruction/timescales little for the directed East constraint, empirically justifying the lean contiguous basis; anchored-contiguous is too starved at small l.

**Downward closure** (include all sub-clusters up to the chosen order) is required so that {0,1} and {-1,+1} encodings span the same function space (they differ by an invertible affine transformation that needs lower-order terms present). Keep the dictionary downward-closed in p.

**Representational capacity vs dynamical closure (do not conflate).** Walsh completeness at order p is a statement about *representational capacity* — what functions of the local window the dictionary can express. It is NOT a claim that the *dynamics* closes at order p. The generator couples orders (a single occupation evolves into a pair, a pair into a triple — a BBGKY-style hierarchy; see below). Whether the dynamics effectively closes at low order is an empirical question, and arguably one of the interesting things the p-sweep measures.

**Why higher order matters (generator non-closure).** The single-site (order-1) dictionary is not closed under the generator. Working out the generator on a single occupation gives

    d/dt E[s_j] = e^(-1/T) E[s_{j+1}] - (1 + e^(-1/T)) E[s_j s_{j+1}]

so single-site evolution depends on the pair quantity E[s_j s_{j+1}], outside the order-1 span; the next order couples to triples, and so on. A truncated dictionary cuts this hierarchy and projects the remainder onto what it has. Concrete falsifiable prediction at low T (large l): the low-order dictionary under-resolves the slow cooperative modes; raising p (and J) should push resolved modes toward Re(mu) = 0, lower slow-mode residuals, and bring the reconstructed decay into closer agreement with tau_ACF.

---

## Critical Implementation Detail: Average Products Over Walkers

The single most common implementation error to avoid.

The Koopman operator evolves expectations:

    E[g(s(t+dt)) | s(0)] ~ K * E[g(s(t)) | s(0)]

For product observables this means

    E[s_i(t) s_j(t) | s(0)]   !=   E[s_i(t) | s(0)] * E[s_j(t) | s(0)]

The difference is the connected correlation, which carries the cooperative physics — the entire signal the higher-order dictionary exists to capture.

**Correct pipeline:**
1. Evaluate each monomial chi_S on each walker's binary configuration at each time -> products computed per walker.
2. Average those products over the W walkers -> estimates of E[chi_S | s(0)].
3. Then apply analytic equilibrium centering/scaling.

**Wrong pipeline:** computing single-site propensities p_i(t) = (1/W) sum_w s_i^w(t) first and then forming products of propensities. This omits the connected correlation entirely and silently destroys the cooperative signal.

---

## Centering and Scaling (analytic, no equilibrium run)

Center and scale each dictionary observable by its **equilibrium** mean and standard deviation — never the empirical window mean.

**Why equilibrium, not window:** subtracting the window mean of a partially-relaxed observable centers it mid-decay, which causes DMD to manufacture spurious oscillatory modes (a known DMD pathology with temporally mean-centered data). Subtracting a constant (the equilibrium value) shifts where the constant-mode amplitude sits and leaves all decaying modes unchanged.

**Closed form (the equilibrium is a product measure, so no simulation is needed).** For any cluster S with |S| spins, the equilibrium spins are i.i.d. Bernoulli(c_eq), so the product is itself Bernoulli with success probability c_eq^|S|:

    <chi_S>_eq = c_eq^|S|
    Var(chi_S)_eq = c_eq^|S| (1 - c_eq^|S|)

The standardized observable is therefore

    chi_S^standardized = ( <chi_S> - c_eq^|S| ) / sqrt( c_eq^|S| (1 - c_eq^|S|) )

where <chi_S> is the walker-averaged product from the pipeline above. This reduces to the existing single-site code, (s - c_eq)/sqrt(c_eq(1-c_eq)), at |S| = 1. Use the analytic value for centering; if there is a systematic finite-size offset in the late-time propensities, a quick check that the empirical late-time mean agrees with c_eq^|S| is worthwhile, but the analytic value is the default.

This transformation is a constant per observable, commutes with walker-averaging, and can be applied at any stage of the pipeline.

**Keep the constant observable {1} in the dictionary even after centering.** It is the trivially conserved Koopman eigenfunction (mu = 0) — necessary for operator closure and conditioning. After centering, its amplitude on each non-constant observable's reconstruction should be near zero, which is correct.

---

## Data Generation Pipeline

For each target temperature T:

1. **Sample n_configs initial configurations from equilibrium at T.** With the pooled per-site representation, few configs are needed (each large-N chain supplies N pooled local samples). Use a long equilibrium pre-thermalization, then take independent snapshots. Chain length N must satisfy N >> l(T); recommend N >= 50 * l, capped by compute budget. All trajectories at a given T must share the same N for pooling.

2. **For each initial configuration, run W walkers** with distinct PRNG seeds, all starting from the same configuration. Walkers must share the initial state exactly and differ only in their stochastic realization. W controls the variance of the expectation estimate; see "Implementation Notes" for the convergence diagnostic.

3. **Sample on a uniform time grid** with timestep dt. DMD assumes fixed-dt snapshot pairs.
   - T_window must be long enough to see the slowest mode of interest. The resolution floor 1/T_window is fundamental (see "Fundamental Limits") and becomes the binding constraint at low T.
   - dt should resolve the fastest dynamics (microscopic timescale ~1).
   - At low T, dynamic range is enormous; prioritize sufficient T_window over fine dt.

4. **Sampling regime:** equilibrium-sampled initial configs run at the same T (NOT a quench). The product-measure equilibrium and the kinetic-constraint dynamics jointly satisfy detailed balance.

**Held-out configurations.** Generating fresh equilibrium configurations is trivial for the East model. Reserve a held-out set (new configs, each with its own W walkers) that are *not* used in the DMD fit, for out-of-sample reconstruction tests (Goal 4). Held-out evaluation requires walker-averaged trajectories (we compare to E[g | s(0)], not a single noisy realization), so new configs must also be run with W walkers.

---

## DMD Pipeline (mirror the Brunton paper)

1. **Build pooled snapshot pairs.** From the standardized local cluster trajectories, stack the per-site, per-config snapshot pairs into joint Psi_X, Psi_Y matrices of shape (D, T_traj * N * (D_t - 1)). Always pool over both sites and configs; do not fit per-trajectory (see Implementation Notes).

2. **Exact DMD with SVD truncation.** Truncate to rank r retaining a target fraction of singular-value energy (start with ~99%). Form the reduced operator A_tilde = U* Y V Sigma^-1 and compute its eigendecomposition. (This is the `fit_exact_dmd` already in the notebooks.)

3. **Continuous-time eigenvalues:** mu_i = log(lambda_i) / dt.

4. **resDMD residual filter.** For each eigenpair, compute the residual eps_i. Keep modes with eps_i < eps_threshold. Start with eps_threshold = 5e-8 (Brunton's value); tune.

5. **Pseudospectra.** Compute eps-pseudospectrum on a grid in the complex mu-plane for visualization (a la Brunton Fig. 4). (`dmd_pseudospectrum_mu` already implements this.)

6. **Mode amplitudes and reconstruction (initial-amplitude least squares — already correct in the notebooks).** Obtain amplitudes by least-squares projection of the initial standardized local state onto the eigenvectors: b = lstsq(Phi, psi0). Do NOT invert Phi. The `lstsq` solve (with singular-value cutoff via rcond) is the regularization that handles ill-conditioned / near-degenerate eigenvectors. Reconstruct as

       g_hat_k(t) ~ sum_i Phi_tilde_k^i b_i(0) e^(mu_i t)

   This is faithful to the Brunton paper, which itself reconstructs from initial mode amplitudes b_i(0). Whole-trajectory amplitude fitting is a possible fallback only if reconstruction quality demands it — not a default, and not needed unless a concrete failure appears.

7. **Relevance filter (optional second stage):** among low-eps modes, keep the minimum number required to reconstruct trajectories to a target relative accuracy (start with 1e-3, Brunton's value). Measure mode contribution as **integrated effect over the window**, not raw |b_i(0)|. **Caution:** do not threshold away slow modes — they carry the scientifically important long-time behavior. Apply the relevance filter mainly to prune the fast/noisy bulk.

**On conditioning.** The ill-conditioned-eigenvector problem that plagued per-site reconstruction in the old full-chain setup was driven largely by translation degeneracy — shift-related directions appearing as near-parallel modes. Pooling over sites dissolves that source by construction. Residual near-degeneracy still occurs whenever two physical timescales are close, which in a glassy system with modes accumulating near the imaginary axis is exactly the regime of interest. The `lstsq` amplitude solve is the correct, paper-faithful machinery for this; it is in the pipeline because near-degenerate slow modes are the physics, not as a band-aid. No bespoke conditioning machinery is needed.

For implementation details (exact-DMD with SVD projection, resDMD residual computation, pseudospectrum evaluation), follow the reference repository. Whether large-D cases require chunked linear algebra (dask) is not yet clear; extend to dask only if memory/runtime demands it.

---

## Phase 0 — Gating: validate the representation before extracting any timescale

This phase is **gating**. Do not extract or trust any tau_DMD, scaling collapse, or spectral-broadening claim until the representation has been validated and sensible (J, p) operating points have been pinned per temperature. Under-resolved spectra produce confidently wrong timescales.

**Reconstruction-quality metric.** Use the average relative L2 error between reconstructed and measured walker-averaged local trajectories over the window, pooled. (Plain pooled average for now; a late-time-weighted variant may be revisited once curves are in hand, since plain L2 can be dominated by the fast early transient and under-weight the slow-mode fidelity that matters for timescale work. Day-to-day specifics of which error to inspect will be directed per session.)

**Gating studies:**

1. **J-sweep (the headline validation).** At fixed modest p (default contiguity mode), sweep J and watch reconstruction quality. Prediction: quality improves with J then **plateaus once J reaches the correlation length l(T)** (static and dynamic length scales coincide in the East model). The plateau location is the cleanest structural confirmation that the local representation is sound.

2. **Temperature dependence of the plateau (stronger confirmation).** The dynamic length scale grows as T falls, so the saturating J should **increase monotonically as T decreases** across the l = 3, 4, 6, 8 (T = 1.443, 0.910, 0.621, 0.514) series. A plateau-J that tracks l(T) ties the representational requirement directly to the physics and is a much stronger result than the plateau alone. This folds in the old "dictionary length scale grows with l" goal as a gating check rather than a standalone headline.

3. **p-sweep.** At fixed J, sweep coupling order p. Prediction: quality also saturates, with the saturating p staying **modest** (facilitation is a low-order effect). The 2D (J, p) saturation picture is richer and more falsifiable than a single curve: J probes how far correlations reach, p probes how high-order they are.

4. **Contiguity-mode comparison (one-off).** At one or two representative temperatures, compare anchored-contiguous vs floating-contiguous vs all-subsets at matched (J, p). Measure whether gapped (all-subsets) terms change reconstruction/timescales. Use the result to *justify* the lean floating-contiguous default empirically (expected: gapped terms add little; anchored is too starved at small l).

**Gate criterion:** representation is accepted once (1)-(2) show a sensible plateau tracking l(T) and a working (J, p) operating point is chosen per temperature. Then proceed to the standing goals.

---

## Standing Goals (post-gate)

### Goal 1 — tau_DMD with super-Arrhenius scaling

Extract tau_DMD across T and compare against:
- **tau_iso** — 1/e-crossing of the isoconfigurational propensity decay (`Isoconfigurational.ipynb`).
- **tau_ACF** — 1/e-crossing of the equilibrium spin autocorrelation (`ACF.ipynb`).

Be explicit about *which* DMD timescale, since the spectrum has multiple natural scales (microscopic, leading activated e^(-1/T), slowest resolved mode). Two definitions to implement and compare:
- **(a) Reconstructed-trajectory 1/e time.** Reconstruct the walker-averaged ACF/propensity from the DMD modes and read off the 1/e crossing. Closest apples-to-apples match with tau_ACF (itself a 1/e crossing).
- **(b) Weighted effective rate.** A weighted average over modes, e.g. tau_eff = sum|b_i| / sum|b_i||Re(mu_i)|. Weights may incorporate amplitude, residual, and integrated effect; explore and report the choice.

**Validation target:** all three should exhibit Bassler scaling log tau ~ A/T^2 with A ~ 1/ln 2 ~ 1.44. They will NOT necessarily agree in absolute value: tau_iso and tau_ACF are 1/e summaries of stretched-exponential decay, while a slowest-mode tau_DMD is the asymptotic tail timescale. The scaling collapse (parallel slopes on log tau vs 1/T^2) is the claim. The reconstructed-ACF 1/e definition (a) is the closest match in absolute value.

### Goal 2 — Extrapolated timescale extraction vs lowered training cutoff

Train the operator on progressively **shorter early-time windows** and test how far the learned operator predicts the long-time tail. The payoff is extrapolating tau to temperatures where direct simulation cannot reach full relaxation. Measure on the predicted reconstructed-trajectory averages using the Phase 0 reconstruction-quality scalar. The resolution floor 1/T_window is the central scientific limit here (see "Fundamental Limits").

Organize as part of a generalization battery (see Goal 4): {train configs, held-out configs} x {full window, truncated window}.

### Goal 3 — Spectral analysis recovering stretched-exponential behavior

Recover the glassy signature in the new representation: accumulation of decaying modes toward Re(mu) = 0 with no clean gap, the direct analog of Brunton's "no-gap" algebraic case. As T decreases, more tiers of the barrier hierarchy populate and the spectrum hugs the imaginary axis (already visible in the single-site series). Quantify via:
- a histogram of modes by integer activation tier j ~ -T ln|Re(mu_i)| (bin in integer j, not in |Re(mu)| — the tiers e^(-k/T) compress geometrically toward zero, so they are evenly spaced only in j), weighted by mode-importance (amplitude / integrated-effect / residual-aware weight, not raw count);
- the density of modes near the imaginary axis.

"Number of populated tiers" or "amplitude-weighted mean tier index" is a candidate data-driven order parameter — the East-model analog of Brunton's eta.

**Note:** the East model is stretched-exponential at all temperatures in the asymmetric limit (always fragile). What changes with T is the *width* of the spectrum and the number of populated barrier tiers, NOT a sharp exponential->stretched transition. Frame results accordingly.

### Goal 4 — Generalization: across configurations and forward in time

The 2x2 battery: **{train configs, held-out configs} x {full window, truncated window}**.
- *Held-out configs* tests whether the operator generalizes across initial conditions (the stringent default validation, not an afterthought).
- *Truncated window* (Goal 2) tests temporal extrapolation.
- The most stringent cell — **held-out configs + truncated window** — is the real prize: predicting the slow decay of a fresh configuration from only early-time training data.

Measure on the predicted reconstructed-trajectory averages using the Phase 0 reconstruction-quality scalar.

---

## Fundamental Limits

- **Resolution floor:** modes with timescale exceeding T_window appear nearly constant over the window and cannot be reliably distinguished from the mu = 0 mode. The conservative systematic error bound is 1/T_window (Brunton's grey bands in their Fig. 5). Binding at low T; check before reading the near-zero accumulation as physics.

- **Spurious positive-Re(mu) eigenvalues:** some DMD eigenvalues will appear with Re(mu) > 0 even after residual filtering — forbidden growing modes that are finite-data artifacts of non-normality. Use the spread of their real parts as an empirical random-error estimate (Brunton's eta error-bar construction).

- **Conditioning at low T:** product observables have equilibrium means c_eq^|S|, tiny at low T. Analytic equilibrium-mean centering and equilibrium-std scaling are essential for numerical stability; without them the SVD truncation behaves erratically.

- **Disambiguate the two causes of a too-fast low-T reconstruction before chasing higher (J, p):** dictionary under-expressiveness (fixable with higher J/p) vs slowest modes lying beyond 1/T_window (fixable only with longer trajectories). They look similar in the reconstructed decay but have different fixes. Confirm tau_ACF lies inside the observation window before attributing a fast-decaying reconstruction to dictionary limits.

- **Memory/runtime:** dictionary size grows with (J, p, contiguity mode); floating-contiguous keeps D roughly linear in J. The pooled snapshot count T_traj * N * (D_t - 1) is large — favor it (it sharpens the fit) but watch memory. Use chunked linear algebra (dask) only if needed.

---

## Implementation Notes

- **Encoding ({0,1} vs {-1,+1}):** with a downward-closed dictionary these are affine-equivalent and produce identical spectra in exact arithmetic. Use {0,1} (matches the existing code) and rely on analytic equilibrium-centering/scaling for conditioning. Do NOT switch encodings expecting more signal — affine equivalence means signal is relocated, not added.

- **Walker count W:** controls variance of dictionary averages. Established diagnostic (`DMD_n_walkers.ipynb`): as W increases, eigenvalue imaginary parts collapse onto the real axis and the resolved spectrum stabilizes. Increase W until the slow modes (the ones that matter) are stable.

- **n_configs:** few are needed because pooling over the N sites of each large-N chain already supplies many statistically-independent local relaxation environments. Spatial extent substitutes for an ensemble of ICs via the (now explicitly exploited) translation invariance. Brunton used 5 ICs in the minimal example and 3 in the oscillator glass. Do not over-invest in this knob; the richer per-site data needs fewer configs than the old full-chain setup. (Held-out evaluation needs *additional* fresh configs beyond the fitting set.)

- **Translation invariance:** the dynamics is statistically translation-invariant. The per-site pooled representation builds this in by construction (every site is a sample of the same local process). This is the central fix relative to the old full-chain representation — see "The Central Change."

- **Pooling vs per-trajectory DMD:** always pool snapshot pairs (over sites and configs) into one fit per temperature. Per-trajectory DMD on a single isoconfigurational trajectory is poorly conditioned (a single relaxation traces out a near-low-dimensional curve in dictionary space and fails to excite the full mode structure).

---

## Parked Extensions (not now)

- **Time-delay / HAVOK embedding.** Stacking time-delayed copies of the local observables is an orthogonal way to enrich the Koopman spectrum (injects more timescales without touching the spatial representation). Likely useful later; explicitly NOT a Phase 0 priority. Park until the spatial (J, p) representation is validated.
- **Polynomial / monomial lift** as a fallback dictionary if the Walsh-Hadamard contiguous construction proves awkward to truncate sensibly.
- **Whole-trajectory amplitude fitting** as a reconstruction fallback only if initial-amplitude least squares proves insufficient.
- **Fragile-to-strong crossover** (intermediate-b model, Buhot-Garrahan section 3.3). The pure East model (b = 0) is fragile at all T; the crossover is a separate study.

---

## What This Project Is Not

- It is not a re-derivation of DMD theory. Use the Brunton paper and reference implementation; deviate only where the binary/stochastic/spatial setting requires it.
- It is not a study of the fragile-to-strong crossover (see Parked Extensions).
- It is not a quench study. All data is equilibrium-sampled and run at the same T. Quench dynamics violate the equilibrium length-distribution stationarity that underpins the relaxation timescale predictions.
- It is not the old full-chain representation. If you find yourself building a single R^(N+1) state vector for the whole chain and pooling over a handful of trajectories, stop — that representation was retired for the reasons in "The Central Change."