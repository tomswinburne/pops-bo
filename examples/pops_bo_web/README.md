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

The true response is

$$f_\text{true}(x) = 0.3x + \sin(2\pi x/P + 0.3) + \delta\sin(4\pi x/P + 0.7),\qquad x\in[0,1],$$

The surrogate knows the fundamental period but omits the harmonic:

$$f_\theta(x) = \theta_0 + \theta_1 x + \theta_2\sin(2\pi x/P) + \theta_3\cos(2\pi x/P).$$

Setup:
- 8 initial points, stratified-random in x ∈ [0, 0.5] by default.
- Gaussian observation noise σ = 0.05.
- 401-point candidate grid shared by all methods.
- 15 BO iterations.
- Per seed, every strategy sees the same initial design and the same noise draws.

**Period.** The spec value is P = 0.45, giving about 2.2 periods on [0, 1]. We also report P = 0.2 (5 periods), a transparent adjustment explained under Results.

## Method (mapping to Lee et al.)

**POPS ensemble (Eq. 6).** The ridge loss is $L(\theta)=\sum_n(\phi_n\cdot\theta-y_n)^2+\lambda|\theta|^2$ with λ = 10⁻⁴. Each observation i gets one member,
$\theta_i=\arg\min L$ subject to $\phi_i\cdot\theta=y_i$. Two variants:

- **Default, exact.** The analytic rank-one correction
  $\theta_i=\theta_*+A^{-1}\phi_i\,e_i/(\phi_i^\top A^{-1}\phi_i)$, where $A=\Phi^\top\Phi+\lambda I$ and $e_i=y_i-\phi_i\cdot\theta_*$. This is the same pointwise correction used in `popsregression`.
- **Eq. 6 approximation.** Refit with 10× loss weight on point i. Select it with `popsVariant: 'weighted'` or "POPS construction" in the UI. Results are indistinguishable from the exact variant (see `results.md`).

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

## Results (40 seeds; full tables in `results.md`)

Median best-measured regret, with mean in parentheses:

| | iter 3 | iter 15 |
|---|---|---|
| **P=0.45, δ=0.12:** POPS+PES / greedy / GP-RBF / GP-periodic | 0.020 / 0.020 / 0.058 / 0.002 | 0.020 (0.022) / 0.020 (0.034) / 0.000 / 0.000 |
| **P=0.45, δ=0.3:** POPS+PES / greedy / GP-RBF | 0.109 / 0.128 / 0.139 | 0.091 (0.097) / 0.119 (0.153) / 0.001 |
| **P=0.2, δ=0.12:** POPS+PES / greedy / GP-RBF | 0.023 / 0.023 / 0.139 | 0.023 / 0.023 / 0.001 |
| **δ=0 (both P):** all parametric | ≈0 | ≈0 |

Why P = 0.2 is also reported:
- At P = 0.45 there are only ~2 periods to explore, so a stationary GP finds the far peak within about 5 experiments. That leaves little scarce-data window.
- The regret floor from the model's location bias scales as curvature × shift² ∝ (1/P²)·P², so it is independent of P.
- The GP's exploration cost grows with the number of periods.
- P = 0.2 therefore widens the window in which a structured model matters, without favouring POPS in any other way. Both settings are reported.

### Answer to the principal question

Does representing model-form uncertainty through POPS improve experimental selection beyond simply using a structured surrogate? In this toy, only modestly, and only when misspecification is large.

1. **The structured surrogate extrapolates.** With data in [0, 0.5], every parametric method locates the far peak at once. The RBF GP needs about 5 (P=0.45) or 8–10 (P=0.2) experiments. This is the parametric inductive bias, and it accounts for nearly all of the early advantage.
2. **POPS tracks model-form error, which is also a floor.**
   - The ensemble collapses at δ = 0 and widens with δ.
   - The same misspecification biases the optimum location of every member, and no experiment within the model class can remove that bias.
   - Parametric methods plateau at ≈0.02 (δ=0.12) and ≈0.1 (δ=0.3) regret, and GP-RBF overtakes them after about 10 iterations.
   - The POPS spread in x\* signals this inadequacy but is narrower than the bias.
3. **PES targets x\* disagreement, not predictive variance.** The acquisition plot overlays the POPS sd to show this.
   - Versus greedy θ\*: tied at δ ≤ 0.12. Better at δ = 0.3 (median 0.091 vs 0.119, mean 0.097 vs 0.153), mainly by removing seeds where greedy keeps re-measuring a wrong point.
   - POPS + UCB performs about the same as PES here. The gain is attributable to POPS uncertainty more than to the entropy criterion.
   - Paper-literal PES (`pops-pes-raw`) has the same recommendation regret but poor best-measured regret, because it never stops exploring.
4. **Control.** The periodic-kernel GP, given the same period knowledge, wins every panel: structural prior, no fixed functional form, no bias floor. Here the prior is expressible as a kernel. For mechanistic models like the kinetic networks in Lee et al. it usually is not, which is where POPS-based BO has a clearer role.
5. **Limitation.** With initial data spanning less than one period (x ∈ [0, 0.35]), all 8 members can extrapolate the wrong trend and agree on a wrong peak. Then H[x\*] ≈ 0 and PES has nothing to resolve. POPS only reflects misfit that the residuals reveal. Try it with "Initial 8 points" in the UI.
