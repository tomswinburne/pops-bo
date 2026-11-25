# pops-bo

POPS-based Bayesian optimization.

**Interactive demo:** <https://tomswinburne.github.io/pops-bo/examples/pops_bo_web/>
([algorithm notes, PDF](https://tomswinburne.github.io/pops-bo/examples/pops_bo_web/docs/pops_bo_algorithm.pdf))

- [`examples/pops_bo_web`](examples/pops_bo_web/): POPS ensembles with data-conditioned approximate
  predictive entropy search (Lee et al. 2026, [doi:10.26434/chemrxiv.15004837/v2](https://doi.org/10.26434/chemrxiv.15004837/v2)),
  compared with POPS + EI, POPS + UCB, a greedy point estimate and GP-EI baselines, with adjustable model
  complexity and 40-seed validation.

The site is deployed by `.github/workflows/jekyll-gh-pages.yml` on every push to `main`.
