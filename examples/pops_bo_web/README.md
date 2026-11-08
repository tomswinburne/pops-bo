# POPS Bayesian optimization: interactive demo

An interactive, dependency-free web demo of POPS-based Bayesian optimization with
approximate predictive entropy search (PES) over the location of the optimum. It
follows Lee et al. (2026) and compares against GP expected improvement and a
parametric greedy ablation on a periodic 1-D toy problem.

> Lee, J., Augustine, L. J., Yang, P., and Perez, D. (2026). *Efficient Bayesian
> Optimization of Separation Chemistry in the Scarce-Data Regime Using Kinetic
> Reaction Networks.* ChemRxiv, preprint, July 13, 2026.
> https://doi.org/10.26434/chemrxiv.15004837/v2

## Run

```bash
# open directly (no server or build step needed)
xdg-open index.html            # or: open index.html / double-click

# regenerate the seeded comparison (writes results.json, results.js, results.md)
node validate.js               # 40 seeds, 15 iterations, ~5 min
node validate.js --seeds 10 --periods 0.45 --deltas 0.12   # quick
```

| file | contents |
|---|---|
| `pops_bo.js` | POPS fit, PES, GP + EI, greedy, campaign driver (UMD: browser and Node) |
| `index.html` | single-page UI: main, x\*-distribution, acquisition and convergence plots, seed-averaged results |
| `validate.js` | seeded validation → `results.json` / `results.js` (loaded by the page) / `results.md` |

## Problem

The truth is a periodic response with a decaying harmonic series, observed with Gaussian noise σ = 0.05:

$$f_\text{true}(x) = 0.3x + \sin(2\pi x/P + 0.3) + \delta\sum_{k=2}^{8}\rho^{k-2}\sin(2\pi k x/P + \phi_k),\qquad x\in[0,1],$$

with $\phi_2 = 0.7$ and $\phi_k = 0.7 + 1.9(k-2)$.

- **ρ = 0** recovers the spec function, which has only the 2nd harmonic.
- **ρ = 0.6** is the default. Every finite-K surrogate is then misspecified, by an amount that shrinks with K.
- **Defaults:** δ = 0.3, P = 0.45 (spec value); P = 0.2 (5 periods) is also reported.

The surrogate has complexity K (number of harmonics, $p = 2 + 2K$):

$$f_\theta(x) = \theta_0 + \theta_1 x + \sum_{k=1}^{K}\big[a_k\sin(2\pi kx/P) + b_k\cos(2\pi kx/P)\big].$$

K = 1 is the spec surrogate. K is set by a slider (1–6), or chosen at every step by the Bayesian evidence (the default). Every method starts from 8 points, stratified-random in x ∈ [0, 0.5]. The candidate grid has 401 points and campaigns run 15 BO iterations. For each seed, every strategy sees the same initial design and the same noise draws.

## Method (mapping to Lee et al.)

**POPS ensemble (Eq. 6).** The ridge loss is $L(\theta)=\sum_n(\phi_n\cdot\theta-y_n)^2+\lambda|\theta|^2$ with λ = 10⁻⁴. Each observation i gets one member,
$\theta_i=\arg\min L$ subject to $\phi_i\cdot\theta=y_i$. Two variants:

- **Default, exact.** The analytic rank-one correction
  $\theta_i=\theta_*+A^{-1}\phi_i\,e_i/(\phi_i^\top A^{-1}\phi_i)$, where $A=\Phi^\top\Phi+\lambda I$ and $e_i=y_i-\phi_i\cdot\theta_*$. This is the same pointwise correction used in `popsregression`.
- **Eq. 6 approximation.** Refit with 10× loss weight on point i. Select it with `popsVariant: 'weighted'` or "POPS construction" in the UI. Results are indistinguishable from the exact variant (see `results.md`).

**Epistemic term.** The ensemble is POPS ⊕ Bayesian-ridge epistemic draws, as in `popsregression`, whose predictive variance is misspecification + epistemic.

- The prior is $\theta\sim\mathcal N(0,\tau^2 I)$ with τ = 1, so the ridge penalty is $\lambda = \sigma_n^2/\tau^2$.
- There are 64 members: member m is $\theta_{m \bmod N} + \xi_m$, with $\xi_m\sim\mathcal N(0,\sigma_n^2A^{-1})$.
- Epistemic children share their POPS parent's likelihood weight, so they are not double-counted.
- Without this term, a model with $p \ge N$ interpolates the data, the POPS corrections vanish, and complex models look spuriously certain.

**Model selection.** K is chosen by the log evidence
$\log p(y\mid K) = -\frac{y^\top y - y^\top\Phi\theta_*}{2\sigma_n^2} - \tfrac12\log|A| + \tfrac p2\log\lambda - \tfrac N2\log 2\pi\sigma_n^2$.
This was checked against the direct $\mathcal N(0,\sigma_n^2 I+\tau^2\Phi\Phi^\top)$ density. Leave-one-out was also tried; it drifted to over-complex K.

**Approximate PES (Eqs. 7–12).**

1. For each member, $x^*_j=\arg\max_{\text{grid}} f_{\theta_j}$ (Eq. 7).
2. Base log-weights are $\ell_j=-\sum_n r_{jn}^2/2\sigma^2$, the Gaussian likelihood of the observed data under member j.
3. σ² is the residual variance of the θ\* fit, $\text{RSS}/(N-4)$, floored at the noise variance.
4. At candidate x, each member's prediction is a hypothetical outcome $y_k=f_{\theta_k}(x)$ (Eq. 8).
5. Augmented-data likelihood: $\ell'_j=\ell_j-(f_{\theta_j}(x)-y_k)^2/2\sigma^2$, normalised with log-sum-exp (Eqs. 9–10).
6. Weighted $\mathrm{Var}(x^*)$ (Eq. 11), then $H\approx\tfrac12\log(2\pi e[\mathrm{Var}+\epsilon])$.
7. Average over outcomes k with weights $w^0_k$ and measure $\arg\min_x\mathbb E[H]$ (Eq. 12).

During acquisition only the weights change. POPS is refitted after each real measurement.

**Our additions, labelled in code and UI.**
- ε = 0.01², the x\* resolution we care about.
- If the best expected gain is < 0.05 nats (or all candidates are tied), measure the highest predicted mean. This generalises the requested tie-break.
- The literal form is kept as `pops-pes-raw`, with ε = Δx²/12 and no threshold.

**Baselines.**
- `gp-ei`: zero-mean GP on standardised y with an RBF kernel. Lengthscale and nugget come from a grid search on the log marginal likelihood, with signal variance profiled out and noise variance ≥ ½σ². Uses EI against the best posterior mean at measured points.
- `gpper-ei`: the same with a periodic kernel at the known P plus a linear term. This control separates the value of the structural prior from POPS.
- `greedy`: argmax of $f_{\theta_*}$, the uncertainty ablation.
- `pops-ucb`: POPS mean + 2·sd, a response-variance control.

**Note on sources.** The paper could not be downloaded in the environment where this was built. The equation mapping above follows the specification we were given (Eqs. 6–12, Figs. 3–4); check the numbering against the published version.

## Results (40 seeds; full tables in `results.md`, regenerate with `node validate.js`)

Best-measured regret at P = 0.45, δ = 0.3, ρ = 0.6. The first five rows are POPS + PES with fixed K.

| surrogate | median @3 | median @15 | mean @15 |
|---|---|---|---|
| K = 1 | 0.031 | 0.031 | 0.032 |
| K = 2 | 0.013 | 0.008 | 0.011 |
| K = 4 | 0.051 | 0.001 | 0.002 |
| K = 6 | 0.125 | 0.003 | 0.005 |
| K by evidence | 0.014 | 0.004 | 0.010 |
| greedy θ\*, K by evidence | 0.022 | 0.011 | 0.048 |
| POPS + UCB, K by evidence | 0.022 | 0.000 | 0.013 |
| GP-RBF + EI | 0.139 | 0.000 | 0.001 |
| GP-periodic (known P) + EI | 0.003 | 0.000 | 0.002 |

### Answer to the principal question

1. **The structured surrogate extrapolates.** From 8 points in [0, 0.5], the parametric methods find the far peak immediately. GP-RBF needs about 5 experiments at P = 0.45 and 8–10 at P = 0.2. This advantage comes from the inductive bias.
2. **Complexity trades early certainty for late accuracy.**
   - Low K: narrow initial POPS envelope, but a regret plateau set by the omitted harmonics.
   - High K: wide initial (epistemic) envelope that keeps shrinking.
   - The evidence-chosen K grows with N and tracks the lower envelope of the fixed-K curves.
   - The complexity panel in the UI shows envelope, true error and evidence against N for every K on the current campaign.
3. **POPS uncertainty vs the point estimate.**
   - Effect size: with evidence-chosen K at δ = 0.3, POPS + PES cuts mean regret at iteration 15 by about 4–5×: 0.010 vs 0.048 at P = 0.45 and 0.011 vs 0.036 at P = 0.2. The gain comes from removing seeds where greedy stays locked on a biased optimum.
   - POPS + UCB performs about as well as PES, so most of the gain comes from the ensemble, not from the entropy criterion. At δ ≤ 0.12 the gaps are small.
   - Paper-literal PES (`pops-pes-raw`) keeps exploring and has much worse best-measured regret.
4. **Controls.** GP-RBF catches up by about 10 iterations. The periodic-kernel GP with known P is the strongest method throughout, because it has the structural prior without a fixed form. In this toy the prior is expressible as a kernel; for mechanistic models like the kinetic networks in Lee et al. it generally is not.
5. **Limitations.**
   - BO data cluster near the optimum, so the global fit (and the evidence for higher K) improves slowly once the method starts exploiting.
   - With initial data spanning less than one period, the members can agree on a wrong peak. Then H[x\*] ≈ 0 and PES has nothing to resolve.

## Viewing on a phone

Open the published artifact link in the Claude app or a phone browser. For a public URL, enable GitHub Pages on this repository and serve `examples/pops_bo_web/`. The page is static and needs no build step.
