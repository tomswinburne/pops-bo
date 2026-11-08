#!/usr/bin/env node
/*
 * Seeded validation of every strategy, written to results.json / results.js
 * (results.js is loaded by index.html so the page works from file://).
 *
 *   node validate.js [--seeds 40] [--iters 15] [--noise 0.05] [--periods 0.45,0.2]
 *                    [--deltas 0,0.12,0.3] [--region 0,0.5]
 *
 * All strategies share, per seed, the same initial design and the same noise draws.
 */
const fs = require('fs'), path = require('path');
const B = require('./pops_bo.js');

const args = process.argv.slice(2);
const opt = (k, d) => { const i = args.indexOf('--' + k); return i >= 0 ? args[i + 1] : d; };
const nSeeds = +opt('seeds', 40), iters = +opt('iters', 15), noise = +opt('noise', 0.05);
const periods = opt('periods', '0.45,0.2').split(',').map(Number);
const deltas = opt('deltas', '0,0.12,0.3').split(',').map(Number);
const region = opt('region', '0,0.5').split(',').map(Number);
const strategies = opt('strategies', Object.keys(B.STRATEGIES).join(',')).split(',');
const gridN = +opt('grid', 401);
const grid = B.makeGrid(gridN);

const res = {
  meta: { nSeeds, iters, noise, region, gridN, nInit: 8, popsVariant: 'exact', eps: 1e-4, gainTol: 0.05, epistemic: true, nEns: 64, tau: 1,
          strategies: Object.fromEntries(strategies.map(s => [s, B.STRATEGIES[s]])), generated: new Date().toISOString() },
  runs: {},
};
const f3 = v => v.toFixed(3);
const md = [];

function summarise(R) {
  const T = R[0].length, q = p => Array.from({ length: T }, (_, t) => B.quantile(R.map(r => r[t]), p));
  const mean = Array.from({ length: T }, (_, t) => R.reduce((a, r) => a + r[t], 0) / R.length);
  return { median: q(0.5), q25: q(0.25), q75: q(0.75), mean };
}

for (const P of periods) for (const delta of deltas) {
  const key = `P=${P},delta=${delta}`;
  res.runs[key] = { P, delta, strategies: {} };
  const head = `\n### P = ${P}, δ = ${delta}\n\n| strategy | best-obs. regret median [IQR] @3 | @5 | @10 | @15 | mean @15 | recommendation regret median @15 |\n|---|---|---|---|---|---|---|`;
  console.log(head); md.push(head);
  for (const s of strategies) {
    const R = [], PR = [];
    for (let seed = 1; seed <= nSeeds; seed++) {
      const c = B.campaign({ strategy: s, delta, noise, seed, iters, grid, region, period: P });
      R.push(c.regret); PR.push(c.predOptRegret);
    }
    const S = summarise(R), SP = summarise(PR);
    res.runs[key].strategies[s] = { best: S, rec: SP };
    const cell = t => `${f3(S.median[t])} [${f3(S.q25[t])}, ${f3(S.q75[t])}]`;
    const row = `| ${B.STRATEGIES[s]} | ${cell(3)} | ${cell(5)} | ${cell(10)} | ${cell(iters)} | ${f3(S.mean[iters])} | ${f3(SP.median[iters])} |`;
    console.log(row); md.push(row);
  }
}

// POPS construction variant check (exact rank-one vs Lee et al. Eq. 6 10x weighting)
const vh = `\n### POPS construction variant (POPS + PES, P = ${periods[0]}, δ = 0.12)\n\n| variant | best-obs. median @5 | @15 | mean @15 |\n|---|---|---|---|`;
console.log(vh); md.push(vh);
for (const v of ['exact', 'weighted']) {
  const R = [];
  for (let seed = 1; seed <= nSeeds; seed++) R.push(B.campaign({ strategy: 'pops-pes', delta: 0.12, noise, seed, iters, grid, region, period: periods[0], popsVariant: v }).regret);
  const S = summarise(R);
  const row = `| ${v} | ${f3(S.median[5])} | ${f3(S.median[iters])} | ${f3(S.mean[iters])} |`;
  console.log(row); md.push(row);
  res.variantCheck = res.variantCheck || {}; res.variantCheck[v] = S;
}

// Model complexity: POPS + PES with K = 1..4 harmonics and evidence-selected K,
// against greedy with evidence-selected K and GP-RBF. K = 2 contains f_true exactly.
res.complexity = {};
const KS = [1, 2, 3, 4, 'auto'];
for (const P of periods) for (const delta of deltas.filter(d => d > 0)) {
  const key = `P=${P},delta=${delta}`;
  res.complexity[key] = { P, delta, series: {} };
  const head = `\n### Complexity, P = ${P}, δ = ${delta} (best-measured regret)\n\n| surrogate | median @1 | @3 | @5 | @15 | mean @5 | mean @15 | recommendation median @15 |\n|---|---|---|---|---|---|---|---|`;
  console.log(head); md.push(head);
  const rows = KS.map(K => ['pops-pes', K]).concat([['greedy', 'auto'], ['gp-ei', 1]]);
  for (const [s, K] of rows) {
    const name = s === 'gp-ei' ? 'GP-RBF + EI' : `${s === 'greedy' ? 'Greedy' : 'POPS + PES'}, K = ${K === 'auto' ? 'auto (evidence)' : K}`;
    const id = s === 'gp-ei' ? 'gp-ei' : `${s}:K=${K}`;
    if (res.runs[key] && s === 'gp-ei') { // reuse
      const S = res.runs[key].strategies['gp-ei'];
      res.complexity[key].series[id] = { name, best: S.best, rec: S.rec };
    } else {
      const R = [], PR = [];
      for (let seed = 1; seed <= nSeeds; seed++) {
        const c = B.campaign({ strategy: s, delta, noise, seed, iters, grid, region, period: P, K });
        R.push(c.regret); PR.push(c.predOptRegret);
      }
      res.complexity[key].series[id] = { name, best: summarise(R), rec: summarise(PR) };
    }
    const S = res.complexity[key].series[id].best, SP = res.complexity[key].series[id].rec;
    const row = `| ${name} | ${f3(S.median[1])} | ${f3(S.median[3])} | ${f3(S.median[5])} | ${f3(S.median[iters])} | ${f3(S.mean[5])} | ${f3(S.mean[iters])} | ${f3(SP.median[iters])} |`;
    console.log(row); md.push(row);
  }
}

fs.writeFileSync(path.join(__dirname, 'results.json'), JSON.stringify(res));
fs.writeFileSync(path.join(__dirname, 'results.js'), '// generated by validate.js — do not edit\nwindow.POPS_RESULTS = ' + JSON.stringify(res) + ';\n');
fs.writeFileSync(path.join(__dirname, 'results.md'),
  `# Seeded validation (${nSeeds} seeds, ${iters} iterations, noise σ = ${noise}, initial region [${region}], ${gridN}-point grid)\n\nGenerated by \`node validate.js\`. Regret = f(x_opt) − max f_true over measured x (noise-free). Recommendation regret = regret of the strategy's predicted optimum after the final measurement.\n` + md.join('\n') + '\n');
console.log('\nwrote results.json, results.js, results.md');
