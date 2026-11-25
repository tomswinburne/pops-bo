/*
 * pops_bo.js — POPS ensemble + approximate predictive entropy search (PES),
 * with GP-EI and parametric-greedy baselines, on a 1-D periodic toy problem.
 *
 * Runs unchanged in the browser (window.PopsBO) and in Node (require).
 * No dependencies.
 *
 * Methodology follows Lee, Augustine, Yang & Perez (2026), ChemRxiv,
 * doi:10.26434/chemrxiv.15004837/v2, Eqs. 6-12 (see README.md).
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.PopsBO = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  // ------------------------------------------------------------------ RNG
  function mulberry32(seed) {
    let a = (seed >>> 0) || 1;
    const rng = function () {
      a = (a + 0x6D2B79F5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
    rng.normal = function () {
      let u = 0, v = 0;
      while (u === 0) u = rng();
      v = rng();
      return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    };
    return rng;
  }

  // ------------------------------------------------------- small lin. alg.
  function cholesky(A) { // A: array of rows (n x n), SPD. returns lower L or null
    const n = A.length, L = [];
    for (let i = 0; i < n; i++) L.push(new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let s = A[i][j];
        for (let k = 0; k < j; k++) s -= L[i][k] * L[j][k];
        if (i === j) {
          if (s <= 0) return null;
          L[i][i] = Math.sqrt(s);
        } else L[i][j] = s / L[j][j];
      }
    }
    return L;
  }
  function cholSolve(L, b) { // solve L L^T x = b
    const n = L.length, y = new Float64Array(n), x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = b[i];
      for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
      y[i] = s / L[i][i];
    }
    for (let i = n - 1; i >= 0; i--) {
      let s = y[i];
      for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
      x[i] = s / L[i][i];
    }
    return x;
  }
  function forwardSolve(L, b) {
    const n = L.length, y = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let s = b[i];
      for (let k = 0; k < i; k++) s -= L[i][k] * y[k];
      y[i] = s / L[i][i];
    }
    return y;
  }
  const dot = (a, b) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * b[i]; return s; };
  function logSumExp(v) {
    let m = -Infinity;
    for (const x of v) if (x > m) m = x;
    if (!isFinite(m)) return m;
    let s = 0;
    for (const x of v) s += Math.exp(x - m);
    return m + Math.log(s);
  }
  function softmax(logw) {
    const lse = logSumExp(logw);
    return logw.map(l => Math.exp(l - lse));
  }

  // ------------------------------------------------------------- problem
  // Problem period P (spec default 0.45). Mutable so the same module serves both
  // configurations; set via setPeriod() / cfg.period (all calls are synchronous).
  let PERIOD = 0.45;
  function setPeriod(P) { if (P != null) PERIOD = P; return PERIOD; }
  /*
   * Truth: fundamental + decaying harmonic series (Fourier content up to K_TRUE = 8):
   *   f(x) = 0.3x + sin(w x + 0.3) + delta * sum_{k=2..8} rho^(k-2) sin(k w x + phi_k),  w = 2 pi/P
   * phi_2 = 0.7, phi_k = 0.7 + 1.9 (k-2). rho = 0 recovers the spec function (2nd harmonic only);
   * rho > 0 makes every finite-K surrogate misspecified, with a residual that shrinks with K.
   */
  let RHO = 0.6;
  const K_TRUE = 8;
  function setProblem(cfg) { if (cfg) { setPeriod(cfg.period); if (cfg.rho != null) RHO = cfg.rho; } }
  function fTrue(x, delta) {
    const w = 2 * Math.PI / PERIOD;
    let f = 0.3 * x + Math.sin(w * x + 0.3);
    if (delta) {
      let a = 1;
      for (let k = 2; k <= K_TRUE; k++) {
        if (a < 1e-6) break;
        f += delta * a * Math.sin(k * w * x + 0.7 + 1.9 * (k - 2));
        a *= RHO;
      }
    }
    return f;
  }
  /*
   * Surrogate features with complexity K (number of harmonics), p = 2 + 2K parameters:
   *   f_theta(x) = th0 + th1 x + sum_{k=1..K} [a_k sin(2 pi k x/P) + b_k cos(2 pi k x/P)]
   * K = 1 is the spec model (misspecified for delta != 0); K = 2 contains f_true exactly;
   * K >= 3 is over-parameterised.
   */
  function features(x, K) {
    K = K || 1;
    const w = 2 * Math.PI / PERIOD, f = [1, x];
    for (let k = 1; k <= K; k++) f.push(Math.sin(k * w * x), Math.cos(k * w * x));
    return f;
  }
  const kOf = theta => (theta.length - 2) / 2;

  function makeGrid(n) {
    const g = new Float64Array(n);
    for (let i = 0; i < n; i++) g[i] = i / (n - 1);
    return g;
  }
  function trueOptimum(grid, delta) {
    // dense refinement beyond the grid so regret is measured against the real optimum
    let best = -Infinity, xb = 0;
    const n = 20001;
    for (let i = 0; i < n; i++) {
      const x = i / (n - 1), v = fTrue(x, delta);
      if (v > best) { best = v; xb = x; }
    }
    return { x: xb, f: best };
  }

  // --------------------------------------------------------------- POPS
  /*
   * Ridge loss L(theta) = sum_n w_n (phi_n.theta - y_n)^2 + lambda |theta|^2
   * A = Phi^T W Phi + lambda I,  theta* = A^{-1} Phi^T W y.
   *
   * variant 'exact' (default): theta_i = argmin L s.t. phi_i.theta = y_i
   *    -> rank-one correction  theta_i = theta* + A^{-1} phi_i e_i / (phi_i^T A^{-1} phi_i),
   *       e_i = y_i - phi_i.theta*   (same formula as popsregression's pointwise_correction)
   * variant 'weighted': Lee et al. Eq. 6 approximation, theta_i = argmin L with
   *       the targeted point's loss weight multiplied by `upweight` (default 10).
   */
  function solveRidge(X, y, w, lambda) {
    const NP = X[0].length;
    const A = [], b = new Float64Array(NP);
    for (let a = 0; a < NP; a++) A.push(new Float64Array(NP));
    for (let n = 0; n < X.length; n++) {
      const p = X[n], wn = w ? w[n] : 1;
      for (let a = 0; a < NP; a++) {
        b[a] += wn * p[a] * y[n];
        for (let c = 0; c < NP; c++) A[a][c] += wn * p[a] * p[c];
      }
    }
    for (let a = 0; a < NP; a++) A[a][a] += lambda;
    const L = cholesky(A);
    return { theta: cholSolve(L, b), L };
  }
  function backSolveT(L, b) { // solve L^T x = b
    const n = L.length, x = new Float64Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let s = b[i];
      for (let k = i + 1; k < n; k++) s -= L[k][i] * x[k];
      x[i] = s / L[i][i];
    }
    return x;
  }
  // Ridge penalty = Gaussian prior theta ~ N(0, tau^2 I) with known noise: lambda = sigma_n^2 / tau^2.
  const defaultLambda = (noise, tau) => Math.max((noise || 0) ** 2 / (tau || 1) ** 2, 1e-6);

  /*
   * Bayesian model evidence for complexity K, with prior theta ~ N(0, tau^2 I) and known
   * noise sigma_n (the same prior that defines the ridge penalty lambda = sigma_n^2/tau^2):
   *   log p(y|K) = -[y.y - y.Phi theta*]/(2 sigma^2) - 1/2 log|A| + p/2 log(lambda) - N/2 log(2 pi sigma^2)
   * with A = Phi^T Phi + lambda I. The Occam factor penalises unneeded harmonics; the data-fit
   * term penalises a model whose residuals exceed the noise (misspecification).
   */
  function logEvidence(xs, ys, K, noise, tau) {
    tau = tau || 1;
    const s2 = Math.max(noise * noise, 1e-8), lambda = s2 / (tau * tau);
    const X = xs.map(x => features(x, K)), N = X.length, p = X[0].length;
    const base = solveRidge(X, ys, null, lambda);
    let yy = 0, yPt = 0;
    for (let n = 0; n < N; n++) { yy += ys[n] * ys[n]; yPt += ys[n] * dot(X[n], base.theta); }
    let logdetA = 0; for (let a = 0; a < p; a++) logdetA += 2 * Math.log(base.L[a][a]);
    return -(yy - yPt) / (2 * s2) - 0.5 * logdetA + 0.5 * p * Math.log(lambda) - 0.5 * N * Math.log(2 * Math.PI * s2);
  }
  function selectK(xs, ys, Kmax, noise, tau) {
    const logEv = [];
    for (let K = 1; K <= Kmax; K++) logEv.push(logEvidence(xs, ys, K, noise, tau));
    let best = 0; for (let i = 1; i < logEv.length; i++) if (logEv[i] > logEv[best]) best = i;
    return { K: best + 1, logEv };
  }

  /*
   * Ensemble = POPS members (misspecification) (+) epistemic draws (finite data).
   *   POPS member i: theta_i as above (one per observation).
   *   epistemic (default on): xi ~ N(0, sigma_n^2 A^{-1}), the Bayesian ridge posterior
   *   covariance with known noise. nEns members, member m = theta_{m mod N} + xi_m.
   *   This mirrors popsregression, whose predictive variance is misspecification + epistemic.
   *   With p >= N the POPS corrections vanish (the fit interpolates), so without the
   *   epistemic term a complex model would look spuriously certain.
   */
  function fitPOPS(xs, ys, opts) {
    opts = opts || {};
    const K = opts.K || 1;
    const lambda = opts.lambda != null ? opts.lambda : defaultLambda(opts.noise, opts.tau);
    const variant = opts.variant || 'exact';
    const upweight = opts.upweight || 10;
    const X = xs.map(x => features(x, K)), N = xs.length, NP = X[0].length;
    const base = solveRidge(X, ys, null, lambda);
    const theta = base.theta;
    const resid = ys.map((y, n) => y - dot(X[n], theta));
    const popsMembers = [];
    for (let i = 0; i < N; i++) {
      if (variant === 'weighted') {
        const w = new Float64Array(N).fill(1);
        w[i] = upweight;
        popsMembers.push(solveRidge(X, ys, w, lambda).theta);
      } else {
        const Ainv_phi = cholSolve(base.L, X[i]);
        const h = Math.max(dot(X[i], Ainv_phi), 1e-12);
        const c = resid[i] / h;
        popsMembers.push(Float64Array.from(theta, (t, a) => t + c * Ainv_phi[a]));
      }
    }
    let members = popsMembers, parent = popsMembers.map((_, i) => i);
    const epistemic = opts.epistemic !== false && (opts.noise || 0) > 0;
    if (epistemic) {
      const M = Math.max(opts.nEns || 64, N);
      const rng = mulberry32(9973 * N + 131 * K + 7);
      members = []; parent = [];
      for (let m = 0; m < M; m++) {
        const z = new Float64Array(NP);
        for (let a = 0; a < NP; a++) z[a] = opts.noise * rng.normal();
        const xi = backSolveT(base.L, z);
        const i = m % N;
        members.push(Float64Array.from(popsMembers[i], (t, a) => t + xi[a]));
        parent.push(i);
      }
    }
    // Likelihood scale: residual variance of the base fit (misspecification + noise),
    // floored at the known observation noise variance.
    const dof = Math.max(N - NP, 1);
    let rss = 0;
    for (const r of resid) rss += r * r;
    const sigma2 = Math.max(rss / dof, (opts.noise || 0) ** 2, 1e-10);
    return { theta, members, popsMembers, parent, resid, sigma2, X, ys, K, lambda, epistemic };
  }

  function predictTheta(theta, xg) {
    const K = kOf(theta), out = new Float64Array(xg.length);
    for (let g = 0; g < xg.length; g++) out[g] = dot(features(xg[g], K), theta);
    return out;
  }
  function argmax(a) { let b = 0; for (let i = 1; i < a.length; i++) if (a[i] > a[b]) b = i; return b; }

  // ------------------------------------------------------------ POPS-PES
  /*
   * Approximate PES (Lee et al. Eqs. 7-12).
   *  - member j predicts curve f_j on the grid and optimum x*_j = argmax f_j   (Eq. 7)
   *  - base log-weights: Gaussian log-likelihood of the observed data under member j
   *  - candidate x, hypothetical outcome y_k = f_k(x)  (k over members)       (Eq. 8)
   *  - augmented-data likelihood: l'_j = l_j - (f_j(x)-y_k)^2/(2 sigma^2),
   *    normalised with log-sum-exp                                           (Eqs. 9-10)
   *  - Var_w(x*) under the reweighted ensemble                                (Eq. 11)
   *  - H = 1/2 log(2 pi e (Var + eps)); expected over outcomes k with weights w0_k (Eq. 12)
   *  Only weights change during acquisition; parameters are refitted after the real
   *  measurement.
   */
  function popsState(pops, grid) {
    const M = pops.members.length, N = pops.popsMembers.length;
    const curves = pops.members.map(th => predictTheta(th, grid));
    const xstar = curves.map(c => grid[argmax(c)]);
    // Data log-likelihood of each POPS parent (paper's rule); epistemic children share their
    // parent's weight (they are already posterior draws, so re-weighting them would double count).
    const parentLog = new Float64Array(N);
    if (pops.useDataLik !== false) {
      for (let i = 0; i < N; i++) {
        let s = 0;
        for (let n = 0; n < pops.X.length; n++) {
          const r = pops.ys[n] - dot(pops.X[n], pops.popsMembers[i]);
          s += r * r;
        }
        parentLog[i] = -s / (2 * pops.sigma2);
      }
    }
    const count = new Float64Array(N);
    for (const i of pops.parent) count[i]++;
    const logw0 = Array.from(pops.parent, i => parentLog[i] - Math.log(count[i]));
    const w0 = softmax(logw0);
    return { curves, xstar, logw0, w0, M };
  }

  function weightedVar(xs, w) {
    let m = 0; for (let i = 0; i < xs.length; i++) m += w[i] * xs[i];
    let v = 0; for (let i = 0; i < xs.length; i++) v += w[i] * (xs[i] - m) ** 2;
    return v;
  }
  function gaussEntropy(v, eps) { return 0.5 * Math.log(2 * Math.PI * Math.E * (v + eps)); }

  function pesAcquisition(st, grid, sigma2, eps) {
    const { curves, xstar, logw0, w0, M } = st;
    const H0 = gaussEntropy(weightedVar(xstar, w0), eps);
    const G = grid.length, EH = new Float64Array(G);
    const lw = new Float64Array(M), col = new Float64Array(M), inv2s = 1 / (2 * sigma2);
    for (let g = 0; g < G; g++) {
      for (let j = 0; j < M; j++) col[j] = curves[j][g];
      let acc = 0;
      for (let k = 0; k < M; k++) {
        if (w0[k] < 1e-12) continue;
        const yk = col[k];
        let mx = -Infinity;
        for (let j = 0; j < M; j++) { const d = col[j] - yk; const l = logw0[j] - d * d * inv2s; lw[j] = l; if (l > mx) mx = l; }
        // log-sum-exp normalised weights -> weighted mean / variance of x*
        let Z = 0, m1 = 0, m2 = 0;
        for (let j = 0; j < M; j++) { const w = Math.exp(lw[j] - mx); Z += w; m1 += w * xstar[j]; m2 += w * xstar[j] * xstar[j]; }
        m1 /= Z; const v = Math.max(m2 / Z - m1 * m1, 0);
        acc += w0[k] * gaussEntropy(v, eps);
      }
      EH[g] = acc;
    }
    return { H0, EH, gain: Array.from(EH, e => H0 - e) };
  }

  function ensembleStats(st, grid) {
    const { curves, w0, M } = st;
    const G = grid.length;
    const mean = new Float64Array(G), sd = new Float64Array(G);
    const lo = new Float64Array(G), hi = new Float64Array(G), q25 = new Float64Array(G), q75 = new Float64Array(G);
    const idx = [...Array(M).keys()];
    for (let g = 0; g < G; g++) {
      let m = 0; for (let j = 0; j < M; j++) m += w0[j] * curves[j][g];
      let v = 0; for (let j = 0; j < M; j++) v += w0[j] * (curves[j][g] - m) ** 2;
      mean[g] = m; sd[g] = Math.sqrt(v);
      idx.sort((a, b) => curves[a][g] - curves[b][g]);
      lo[g] = curves[idx[0]][g]; hi[g] = curves[idx[M - 1]][g];
      q25[g] = wquant(idx, curves, g, w0, 0.25);
      q75[g] = wquant(idx, curves, g, w0, 0.75);
    }
    return { mean, sd, lo, hi, q25, q75 };
  }
  function wquant(sortedIdx, curves, g, w, q) {
    let c = 0;
    for (const j of sortedIdx) { c += w[j]; if (c >= q) return curves[j][g]; }
    return curves[sortedIdx[sortedIdx.length - 1]][g];
  }

  // ------------------------------------------------------------------ GP
  /*
   * Zero-mean GP on standardised y. K = s2 (R(l) + g I) [+ s2 b x x' for periodic].
   * Hyperparameters (l, g [, b]) by grid search on log marginal likelihood with
   * s2 profiled analytically:  s2 = y^T C^{-1} y / n,  C = R + gI.
   * Noise: g is bounded below by (noise/std(y))^2 / s2-consistent floor via grid.
   */
  function kernelRBF(l) { return (a, b) => Math.exp(-0.5 * ((a - b) / l) ** 2); }
  function kernelPer(l, b) {
    return (x1, x2) => Math.exp(-2 * Math.sin(Math.PI * Math.abs(x1 - x2) / PERIOD) ** 2 / (l * l)) + b * (x1 - 0.5) * (x2 - 0.5);
  }
  function logspace(a, b, n) { const o = []; for (let i = 0; i < n; i++) o.push(Math.exp(Math.log(a) + (Math.log(b) - Math.log(a)) * i / (n - 1))); return o; }

  function fitGP(xs, ys, opts) {
    opts = opts || {};
    const kind = opts.kind || 'rbf';
    const n = xs.length;
    const mu = ys.reduce((s, v) => s + v, 0) / n;
    let sd = Math.sqrt(ys.reduce((s, v) => s + (v - mu) ** 2, 0) / n) || 1;
    const yn = ys.map(v => (v - mu) / sd);
    const noiseN2 = ((opts.noise || 0) / sd) ** 2; // noise var in standardised units
    const ls = kind === 'rbf' ? logspace(0.02, 1.0, 16) : logspace(0.15, 3.0, 10);
    const gs = logspace(1e-5, 0.3, 9);
    const bs = kind === 'rbf' ? [0] : [0, 1, 10];
    let best = null;
    for (const l of ls) for (const b of bs) {
      const k = kind === 'rbf' ? kernelRBF(l) : kernelPer(l, b);
      const R = [];
      for (let i = 0; i < n; i++) { R.push(new Float64Array(n)); for (let j = 0; j < n; j++) R[i][j] = k(xs[i], xs[j]); }
      for (const g of gs) {
        const C = R.map((r, i) => { const c = Float64Array.from(r); c[i] += g + 1e-9; return c; });
        const L = cholesky(C);
        if (!L) continue;
        const alpha = cholSolve(L, yn);
        const s2 = Math.max(dot(yn, alpha) / n, 1e-12);
        // enforce noise variance >= known observation noise: s2*g >= noiseN2
        if (s2 * g < 0.5 * noiseN2) continue;
        let logdet = 0; for (let i = 0; i < n; i++) logdet += 2 * Math.log(L[i][i]);
        const lml = -0.5 * n * Math.log(s2) - 0.5 * logdet - 0.5 * n;
        if (!best || lml > best.lml) best = { lml, l, g, b, s2, L, alpha, k };
      }
    }
    if (!best) { // fallback: heavy nugget
      const l = 0.2, g = 0.1, k = kind === 'rbf' ? kernelRBF(l) : kernelPer(1, 0);
      const C = []; for (let i = 0; i < n; i++) { C.push(new Float64Array(n)); for (let j = 0; j < n; j++) C[i][j] = k(xs[i], xs[j]) + (i === j ? g : 0); }
      const L = cholesky(C), alpha = cholSolve(L, yn);
      best = { lml: NaN, l, g, b: 0, s2: dot(yn, alpha) / n, L, alpha, k };
    }
    return { ...best, xs, mu, sd, kind };
  }
  function predictGP(gp, grid) {
    const G = grid.length, n = gp.xs.length;
    const mean = new Float64Array(G), std = new Float64Array(G);
    const kv = new Float64Array(n);
    for (let g = 0; g < G; g++) {
      for (let i = 0; i < n; i++) kv[i] = gp.k(grid[g], gp.xs[i]);
      const m = dot(kv, gp.alpha);
      const v = forwardSolve(gp.L, kv);
      const kss = gp.k(grid[g], grid[g]);
      const var_ = Math.max(gp.s2 * (kss - dot(v, v)), 1e-14);
      mean[g] = gp.mu + gp.sd * m;
      std[g] = gp.sd * Math.sqrt(var_); // latent-f std (noise excluded)
    }
    return { mean, std };
  }
  function normPdf(z) { return Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI); }
  function normCdf(z) { // Abramowitz-Stegun 7.1.26 via erf
    const t = 1 / (1 + 0.3275911 * Math.abs(z) / Math.SQRT2);
    const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z / 2);
    return z >= 0 ? 0.5 * (1 + y) : 0.5 * (1 - y);
  }
  function expectedImprovement(pred, incumbent, xi) {
    const ei = new Float64Array(pred.mean.length);
    for (let g = 0; g < ei.length; g++) {
      const s = pred.std[g], d = pred.mean[g] - incumbent - xi;
      if (s < 1e-12) { ei[g] = Math.max(d, 0); continue; }
      const z = d / s;
      ei[g] = d * normCdf(z) + s * normPdf(z);
    }
    return ei;
  }

  // ------------------------------------------------------------ strategies
  const STRATEGIES = {
    'pops-pes': 'POPS + PES',
    'pops-pes-raw': 'POPS + PES (raw: grid-scale ε, no exploit threshold)',
    'pops-ucb': 'POPS + UCB (response-variance control)',
    'greedy': 'Parametric point estimate (greedy)',
    'gp-ei': 'GP (RBF) + EI',
    'gpper-ei': 'GP (periodic+linear, known P) + EI',
  };

  /*
   * One acquisition decision. Returns everything the UI needs to draw.
   * cfg: {strategy, grid, noise, popsVariant, eps, ucbBeta, eiXi, useDataLik}
   */
  function propose(xs, ys, cfg) {
    setProblem(cfg);
    const grid = cfg.grid; let s = cfg.strategy;
    if (s === 'pops-pes-raw') { cfg = { ...cfg, eps: (grid[1] - grid[0]) ** 2 / 12, gainTol: 0 }; s = 'pops-pes'; }
    const out = { strategy: s };
    // Anti-locality: candidates closer than r = minSpacing to any previous experiment are not
    // eligible for the NEXT experiment (the recommendation x_hat is unaffected). A repeat
    // measurement only re-averages noise; under misspecification the nearby points carry the
    // information the model cannot represent. Falls back to all candidates if none are eligible.
    const r = cfg.minSpacing != null ? cfg.minSpacing : 0.005;
    const eligible = new Uint8Array(grid.length).fill(1);
    if (r > 0) {
      for (let g = 0; g < grid.length; g++) for (const x of xs) if (Math.abs(grid[g] - x) < r) { eligible[g] = 0; break; }
      if (!eligible.some(v => v)) eligible.fill(1);
    }
    const pick = a => { let b = -1; for (let g = 0; g < a.length; g++) if (eligible[g] && (b < 0 || a[g] > a[b])) b = g; return b; };
    out.eligible = eligible; out.minSpacing = r;
    if (s.startsWith('pops') || s === 'greedy') {
      const lam = cfg.lambda != null ? cfg.lambda : defaultLambda(cfg.noise, cfg.tau);
      let K = cfg.K || 1;
      if (K === 'auto') { const sel = selectK(xs, ys, cfg.Kmax || 6, cfg.noise, cfg.tau); K = sel.K; out.logEv = sel.logEv; }
      out.K = K;
      const pops = fitPOPS(xs, ys, { variant: cfg.popsVariant, noise: cfg.noise, lambda: lam, K, epistemic: cfg.epistemic, nEns: cfg.nEns });
      pops.useDataLik = cfg.useDataLik;
      const st = popsState(pops, grid);
      const stats = ensembleStats(st, grid);
      const thetaCurve = predictTheta(pops.theta, grid);
      out.pops = pops; out.st = st; out.stats = stats; out.thetaCurve = thetaCurve;
      out.mean = stats.mean;
      if (s === 'greedy') {
        out.acq = thetaCurve; out.acqLabel = 'f_θ*(x) (maximise)';
        out.next = pick(thetaCurve);
        out.predOpt = argmax(thetaCurve);
      } else if (s === 'pops-ucb') {
        const beta = cfg.ucbBeta != null ? cfg.ucbBeta : 2;
        out.acq = Float64Array.from(stats.mean, (m, g) => m + beta * stats.sd[g]);
        out.acqLabel = 'mean + ' + beta + '·sd (maximise)';
        out.next = pick(out.acq);
        out.predOpt = argmax(stats.mean);
      } else {
        // eps: variance floor = squared resolution at which we care about x* (default 0.01^2)
        const eps = cfg.eps != null ? cfg.eps : 1e-4;
        const pes = pesAcquisition(st, grid, pops.sigma2, eps);
        out.pes = pes;
        out.acq = pes.gain; out.acqLabel = 'H[x*] − E[H[x*] | y(x)]  (maximise)';
        let mx = -Infinity, mn = Infinity;
        for (let g = 0; g < grid.length; g++) if (eligible[g]) { const v = pes.gain[g]; if (v > mx) mx = v; if (v < mn) mn = v; }
        out.predOpt = argmax(stats.mean);
        // Converged / tied: every member agrees on x* to within the resolution eps, so no
        // candidate is informative -> exploit the highest predicted mean.
        const tol = cfg.gainTol != null ? cfg.gainTol : 0.05;
        if (mx - mn < 1e-9 || mx < tol) { out.next = pick(stats.mean); out.tieBreak = true; }
        else {
          // ties within tolerance: highest predicted mean among the top acquisition values
          let best = -1;
          for (let g = 0; g < grid.length; g++) {
            if (eligible[g] && pes.gain[g] >= mx - 1e-9 && (best < 0 || stats.mean[g] > stats.mean[best])) best = g;
          }
          out.next = best;
        }
      }
    } else {
      const gp = fitGP(xs, ys, { kind: s === 'gp-ei' ? 'rbf' : 'per', noise: cfg.noise });
      const pred = predictGP(gp, grid);
      // noisy incumbent: best posterior mean at observed locations
      const atObs = predictGP(gp, xs).mean;
      const inc = Math.max(...atObs);
      const xi = (cfg.eiXi != null ? cfg.eiXi : 0.01) * gp.sd;
      const ei = expectedImprovement(pred, inc, xi);
      out.gp = gp; out.pred = pred; out.mean = pred.mean;
      out.acq = ei; out.acqLabel = 'expected improvement (maximise)';
      out.predOpt = argmax(pred.mean);
      out.next = pick(ei);
      if (ei[out.next] < 1e-12) out.next = pick(pred.mean);
    }
    out.xNext = grid[out.next];
    out.xPredOpt = grid[out.predOpt];
    return out;
  }

  function initialDesign(seed, region, n) {
    const rng = mulberry32(seed * 7919 + 17);
    const [a, b] = region;
    const xs = [];
    // stratified (one point per sub-interval) for a reproducible but random spread
    for (let i = 0; i < n; i++) xs.push(a + (b - a) * (i + rng()) / n);
    return xs;
  }

  function observe(x, delta, noise, rng) { return fTrue(x, delta) + noise * rng.normal(); }

  /*
   * Full campaign: same initial design and noise stream for every strategy given a seed.
   */
  /*
   * Campaign state. For a given seed every strategy sees the same initial design and the
   * same noise draws (initial noise, then one draw per BO iteration).
   */
  function startCampaign(cfg) {
    setProblem(cfg);
    const delta = cfg.delta, noise = cfg.noise, iters = cfg.iters || 15;
    const grid = cfg.grid || makeGrid(cfg.gridN || 401);
    const xs0 = initialDesign(cfg.seed, cfg.region || [0, 0.5], cfg.nInit || 8);
    const rng = mulberry32(cfg.seed * 104729 + 3);
    const noiseInit = xs0.map(() => rng.normal());
    const noiseSeq = []; for (let i = 0; i < Math.max(iters, 60); i++) noiseSeq.push(rng.normal());
    const xs = xs0.slice(), ys = xs0.map((x, i) => fTrue(x, delta) + noise * noiseInit[i]);
    const opt = trueOptimum(grid, delta);
    const b0 = Math.max(...xs.map(x => fTrue(x, delta)));
    return { cfg: { ...cfg, grid }, grid, xs, ys, nInit: xs0.length, noiseSeq, t: 0, opt,
             bestTrue: [b0], regret: [opt.f - b0], predOptRegret: [], proposals: [] };
  }
  function proposeNext(state) {
    const p = propose(state.xs, state.ys, state.cfg);
    if (state.predOptRegret.length === state.t) state.predOptRegret.push(state.opt.f - fTrue(p.xPredOpt, state.cfg.delta));
    return p;
  }
  function applyProposal(state, p) {
    const { delta, noise } = state.cfg;
    const x = p.xNext, f = fTrue(x, delta);
    state.xs.push(x); state.ys.push(f + noise * state.noiseSeq[state.t]);
    state.proposals.push(p);
    const b = Math.max(state.bestTrue[state.bestTrue.length - 1], f);
    state.bestTrue.push(b); state.regret.push(state.opt.f - b);
    state.t++;
  }
  function campaign(cfg) {
    const st = startCampaign(cfg);
    for (let t = 0; t < cfg.iters; t++) applyProposal(st, proposeNext(st));
    proposeNext(st); // recommendation after the final measurement
    return { xs: st.xs, ys: st.ys, bestTrue: st.bestTrue, regret: st.regret, predOptRegret: st.predOptRegret, opt: st.opt };
  }

  function quantile(arr, q) {
    const a = Float64Array.from(arr).sort();
    const pos = (a.length - 1) * q, lo = Math.floor(pos), hi = Math.ceil(pos);
    return a[lo] + (a[hi] - a[lo]) * (pos - lo);
  }

  return {
    get PERIOD() { return PERIOD; }, get RHO() { return RHO; }, K_TRUE, setPeriod, setProblem, fTrue, features, makeGrid, trueOptimum, fitPOPS, popsState, pesAcquisition,
    ensembleStats, predictTheta, logEvidence, selectK, defaultLambda, fitGP, predictGP, expectedImprovement, propose,
    initialDesign, observe, campaign, startCampaign, proposeNext, applyProposal, mulberry32, quantile, STRATEGIES, argmax,
  };
});
