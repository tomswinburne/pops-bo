#!/usr/bin/env node
// Writes docs/results_tables.tex from ../results.json (run after validate.js).
const fs = require('fs'), path = require('path');
const R = JSON.parse(fs.readFileSync(path.join(__dirname, '..', 'results.json'), 'utf8'));
const f3 = v => v.toFixed(3);
const esc = s => s.replace(/&/g, '\\&').replace(/θ\*/g, '$\\theta_*$').replace(/≥/g, '$\\ge$');
const names = {
  'pops-pes': 'POPS + PES (conditioned)', 'pops-pes-raw': 'POPS + PES, paper-literal',
  'pops-ucb': 'POPS + UCB', 'greedy': 'Greedy $\\theta_*$', 'gp-ei': 'GP-RBF + EI', 'gpper-ei': 'GP-periodic + EI',
};
const m = R.meta, T = m.iters, out = [];
out.push(`\\noindent Settings: $\\rho=${m.rho}$, $\\sigma_n=${m.noise}$, ${m.nSeeds} seeds, initial region $[${m.region.join(',')}]$, ` +
  `$K$ ${m.K === 'auto' ? 'by evidence' : '= ' + m.K}, conditioning ${m.discrepancy ? 'on' : 'off'}, minimum spacing $r=${m.minSpacing}$.\n`);

for (const key of Object.keys(R.runs)) {
  const run = R.runs[key];
  if (run.delta !== 0.3) continue;
  out.push(`\\paragraph{Strategies, $P=${run.P}$, $\\delta=${run.delta}$.} Best measured regret (median, with mean) and recommendation regret.\n`);
  out.push('\\begin{center}\\small\\begin{tabular}{lrrrrr}\\toprule');
  out.push(`strategy & med.\\ @3 & med.\\ @5 & med.\\ @${T} & mean @${T} & rec.\\ @${T}\\\\\\midrule`);
  for (const [s, d] of Object.entries(run.strategies)) {
    const b = d.best, r = d.rec;
    out.push(`${names[s] || esc(s)} & ${f3(b.median[3])} & ${f3(b.median[5])} & ${f3(b.median[T])} & ${f3(b.mean[T])} & ${f3(r.median[T])}\\\\`);
  }
  out.push('\\bottomrule\\end{tabular}\\end{center}\n');
}
if (R.complexity) {
  const key = Object.keys(R.complexity).find(k => R.complexity[k].delta === 0.3 && R.complexity[k].P === 0.45) || Object.keys(R.complexity)[0];
  const c = R.complexity[key];
  out.push(`\\paragraph{Model complexity, $P=${c.P}$, $\\delta=${c.delta}$.} Best measured regret; the recommendation column shows the model bias.\n`);
  out.push('\\begin{center}\\small\\begin{tabular}{lrrrrr}\\toprule');
  out.push(`surrogate & med.\\ @1 & med.\\ @3 & med.\\ @5 & med.\\ @${T} & rec.\\ @${T}\\\\\\midrule`);
  for (const [, d] of Object.entries(c.series)) {
    const b = d.best, r = d.rec;
    out.push(`${esc(d.name).replace('K = auto (evidence)', '$K$ by evidence').replace(/K = (\d)/, '$K=$1$')} & ${f3(b.median[1])} & ${f3(b.median[3])} & ${f3(b.median[5])} & ${f3(b.median[T])} & ${f3(r.median[T])}\\\\`);
  }
  out.push('\\bottomrule\\end{tabular}\\end{center}\n');
}
if (R.spacing) {
  for (const key of Object.keys(R.spacing)) {
    const sp = R.spacing[key];
    if (sp.delta !== 0.3) continue;
    out.push(`\\paragraph{Anti-locality ablation, $P=${sp.P}$, $\\delta=${sp.delta}$.} Fraction of BO steps that replicate a previous point (within $0.001$) or land within $0.005$ of one.\n`);
    out.push('\\begin{center}\\small\\begin{tabular}{lccrrrr}\\toprule');
    out.push(`strategy & conditioned & $r$ & replicates & within $0.005$ & med.\\ @5 & mean @${T}\\\\\\midrule`);
    for (const d of Object.values(sp.series)) {
      out.push(`${names[d.strategy] ? names[d.strategy].replace(' (conditioned)', '') : d.strategy} & ${d.strategy.startsWith('gp') ? '--' : d.conditioned ? 'yes' : 'no'} & ${d.r} & ${(100 * (d.exact ?? 0)).toFixed(0)}\\% & ${(100 * d.repeat).toFixed(0)}\\% & ${f3(d.best.median[5])} & ${f3(d.best.mean[T])}\\\\`);
    }
    out.push('\\bottomrule\\end{tabular}\\end{center}\n');
  }
}
fs.writeFileSync(path.join(__dirname, 'results_tables.tex'), out.join('\n'));
console.log('wrote docs/results_tables.tex');
