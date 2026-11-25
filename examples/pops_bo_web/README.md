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
node validate.js               # 40 seeds, 15 iterations, ~45 min serial
node validate.js --seeds 10 --periods 0.45 --deltas 0.12   # quick

# parallel: one shard per (P, delta), then merge
node validate.js --periods 0.45 --deltas 0.3 --variant 0 --part /tmp/p1.json   # ... one per config
node merge_results.js /tmp/p1.json /tmp/p2.json ...

# algorithm notes (tables generated from results.json)
cd docs && node make_tables.js && pdflatex pops_bo_algorithm.tex && pdflatex pops_bo_algorithm.tex
```

| file | contents |
|---|---|
| `pops_bo.js` | POPS fit, PES, GP + EI, greedy, campaign driver (UMD: browser and Node) |
| `index.html` | single-page UI: main, x\*-distribution, acquisition and convergence plots, seed-averaged results |
| `validate.js` | seeded validation → `results.json` / `results.js` (loaded by the page) / `results.md` |
| `merge_results.js` | merges sharded `validate.js --part` outputs |
| `docs/pops_bo_algorithm.{tex,pdf}` | algorithm notes: every equation implemented, plus generated results tables |

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

**POPS ensemble (Eq. 6).** The ridge loss is $L(\theta)=\sum_n(\phi_n\cdot\theta-y_n)^2+\lambda|\theta|^2$ with λ = σ_n²/τ² (prior θ ~ N(0, τ²I), τ = 1). Each observation i gets one member,
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

**Conditioning outcomes on nearby data (default; our extension).** As written, approximate PES draws hypothetical outcomes from the members, with no reference to what was already measured near x. We add a discrepancy GP: $y=f_\theta(x)+d(x)+\epsilon$, with $d\sim\mathcal{GP}(0,\tau_d^2k_\ell)$.
- $(\ell,\tau_d^2)$ are fitted to the residuals of the POPS members, so the GP absorbs the misfit POPS asserts.
- Each member's predicted outcome becomes $\mu_j(x)=f_{\theta_j}(x)+k(x)^\top C^{-1}r_j$.
- The outcome-likelihood width becomes $s^2(x)=\tau_d^2-k^\top C^{-1}k+\sigma_n^2$, replacing σ² in steps 5–7.
- Base weights are the GP marginal likelihood of the parent's residuals, and $x^*_j$ comes from the conditioned curves.
- At a measured point all $\mu_j$ coincide up to noise, so a replicate carries no information.

**Resolved-optimum step (our extension).** When the best PES gain is below 0.05 nats, the members agree on x\* to within ε. The step then maximises the expected improvement of the best *measured* value, averaged over members: $\sum_j w_j\,\mathbb E[\max(f_j(x)-f_j(x_{n_j}),0)]$, where $x_{n_j}$ is member j's best measured point.
- The variance is that of the *difference* $d(x)-d(x_{n_j})$, which vanishes as x approaches the incumbent.
- The criterion is set exactly to zero at every measured point, because no measured point can beat the best measured value.
- The same criterion from the first step is the `pops-ei` strategy.

Full equations are in `docs/pops_bo_algorithm.pdf`.

**Our additions, labelled in code and UI.**
- ε = 0.01², the x\* resolution we care about.
- Outcome conditioning and the resolved-optimum step described above. The highest-mean tie-break is kept as `exploit: 'mean'`.
- Optional hard minimum spacing r between experiments, applying to every strategy. The default is r = 0 since the principled fix above; it is a UI slider.
- The literal form is kept as `pops-pes-raw`, with ε = Δx²/12 and no threshold.

**Baselines.**
- `gp-ei`: zero-mean GP on standardised y with an RBF kernel. Lengthscale and nugget come from a grid search on the log marginal likelihood, with signal variance profiled out and noise variance ≥ ½σ². Uses EI against the best posterior mean at measured points.
- `gpper-ei`: the same with a periodic kernel at the known P plus a linear term. This control separates the value of the structural prior from POPS.
- `greedy`: argmax of $f_{\theta_*}$, the uncertainty ablation.
- `pops-ei`: POPS expected improvement of the best measured value from the first step; a value-seeking control on the same ensemble.
- `pops-ucb`: POPS mean + 2·sd, a response-variance control.

**Note on sources.** The paper could not be downloaded in the environment where this was built. The equation mapping above follows the specification we were given (Eqs. 6–12, Figs. 3–4); check the numbering against the published version.

## Results (40 seeds; full tables in `results.md`)

Best-measured regret, P = 0.45, δ = 0.3, ρ = 0.6, K by evidence, outcomes conditioned, r = 0. Medians, with the mean at iteration 15:

| strategy | @3 | @5 | @10 | @15 | mean @15 | recommendation @15 |
|---|---|---|---|---|---|---|
| POPS + PES | 0.022 | 0.004 | 0.000 | 0.000 | 0.000 | 0.001 |
| POPS + EI | 0.008 | 0.001 | 0.000 | 0.000 | 0.000 | 0.003 |
| POPS + UCB | 0.021 | 0.001 | 0.000 | 0.000 | 0.005 | 0.004 |
| greedy θ\* | 0.022 | 0.021 | 0.011 | 0.011 | 0.041 | 0.018 |
| GP-RBF + EI | 0.139 | 0.092 | 0.001 | 0.000 | 0.001 | 0.001 |
| GP-periodic (known P) + EI | 0.003 | 0.002 | 0.000 | 0.000 | 0.002 | 0.003 |
| paper-literal PES | 0.082 | 0.075 | 0.047 | 0.029 | 0.036 | 0.003 |

Replicate measurements at r = 0 (exact repeats of a previous x):

| strategy | conditioned | exact replicates |
|---|---|---|
| POPS + PES, old exploit rule (re-measure predicted optimum) | — | 60–68% |
| POPS + PES, location-EI exploit | no | 1% |
| POPS + PES, location-EI exploit | yes | 0% |
| POPS + EI | yes | 0% |
| POPS + UCB | yes | 49% |
| greedy θ\* | yes | 79% |
| GP-RBF + EI | — | 31% |

### Answer to the principal question

1. **The structured surrogate extrapolates.** POPS methods reach regret ≈0.01–0.02 by iteration 3. GP-RBF needs about 10 (P = 0.45) or 15 (P = 0.2) experiments. The periodic-kernel GP, which shares the structural prior, is competitive throughout.
2. **Representing model-form uncertainty matters; the acquisition function matters less.**
   - Every POPS-ensemble method beats the greedy point estimate (mean @15: 0.000–0.005 vs 0.041).
   - POPS + EI is the best early performer (0.008 vs 0.022 for PES at iteration 3).
   - In this toy, entropy over x\* does not beat EI on the same conditioned ensemble.
3. **Replicates.** The fix that matters is the resolved-optimum criterion: the improvement of the best measured value, which is exactly zero at measured points. Conditioning outcomes on nearby data makes PES consistent but changes regret little here. Methods without the location criterion (UCB, greedy, GP-EI) still replicate unless the minimum-spacing rule is used.
4. **Complexity.** Large K starts with a wide epistemic envelope (K = 6: 0.139 at iteration 3) but all K converge. The evidence-chosen K gives the lowest recommendation regret (0.001) and grows with N.
5. **Limitations.**
   - Gaussian entropy for a multimodal x\*.
   - The discrepancy GP adds a nonparametric component.
   - With initial data spanning less than one period, the members can agree on a wrong peak.

## Web

- **Demo:** <https://tomswinburne.github.io/pops-bo/examples/pops_bo_web/>
- **Algorithm notes:** <https://tomswinburne.github.io/pops-bo/examples/pops_bo_web/docs/pops_bo_algorithm.pdf>

Both are deployed by the Jekyll Pages workflow on `main`. The page is static, with no build step.
