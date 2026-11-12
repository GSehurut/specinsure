# Contributing to specinsure

Thanks for your interest! Contributions of all sizes are welcome: bug reports, documentation, examples, tests and new features. You do not need to be an actuary to help, and you do not need to be a software engineer either. Both perspectives make the project better.

## Ways to contribute

- **Pick up an issue.** Issues labelled [`good first issue`](https://github.com/GSehurut/specinsure/labels/good%20first%20issue) are scoped for newcomers. Comment on the issue to say you are working on it.
- **Report a bug.** Open an issue with a minimal example that reproduces it.
- **Suggest a feature.** Open an issue describing the actuarial or data problem first, before writing code, so we can agree on the approach.
- **Improve the docs or examples.** Clear explanations of the insurance concepts are as valuable as code.

See [ROADMAP.md](ROADMAP.md) for where the project is heading.

## Development setup

```sh
git clone https://github.com/GSehurut/specinsure.git
cd specinsure
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

## Making a change

1. Fork the repository and create a branch from `main` (`git checkout -b add-reinstatements`).
2. Make your change with tests. Bug fixes should include a test that fails without the fix.
3. Run the checks locally (and `python examples/portfolio_walkthrough.py` if you changed public behaviour):
   ```sh
   ruff check .
   ruff format .
   pytest
   ```
4. Open a pull request describing **what** changed and **why**. Link the issue it closes.

## Architecture

The package is layered; a module may import from layers above it in this list, never below:

| Layer | Package | Put code here when it... |
| --- | --- | --- |
| 1 | `core` | is used everywhere: exceptions, type aliases, argument validation, RNG streams |
| 2 | `distributions` | describes or fits a univariate distribution |
| 3 | `simulation` | turns models into year loss tables (`YearLossTable`, `SimulationResult`) |
| 3 | `risk` | measures risk on arrays of simulated outcomes (no knowledge of treaties) |
| 4 | `reinsurance` | transforms per-loss arrays into ceded amounts, or computes layer metrics |
| 4 | `reserving` | projects claims triangles to ultimate or measures reserve uncertainty |
| 5 | `pricing` | turns experience, exposure or simulations into a loss cost or premium |
| 5 | `portfolio` | combines lines of business, their reinsurance and premiums |
| 6 | `optimisation` | chooses decisions (line weights, structures) using the layers below |
| 6 | `reporting` | formats results; nothing else should import it |

Conventions that keep the numbers right:

- **Losses are positive** in `risk`; underwriting results are positive for profit.
- **Annual terms are applied per simulated year** via `year_index`, never across the whole simulation.
- **Explicit randomness:** functions take a `seed` (`SeedLike`); use `core.random.spawn_rngs` for independent components.
- **Validate at the boundary** with `core.validation` helpers, raising `ValidationError`.
- **Prefer closed-form tests:** check simulated results against formulas (compound Poisson mean, layer expected value from limited expected values) with a tolerance based on the Monte Carlo standard error.

## Guidelines

- **Correctness first.** This is financial and actuarial code. Cite a reference (textbook, paper, market standard) for any non-trivial formula, in the docstring or the PR.
- **Small, focused pull requests** are reviewed faster than large ones.
- **Type hints and docstrings** on public functions.
- **No real or confidential data.** Use synthetic data in tests and examples only.
- **Be kind.** Everyone taking part is expected to follow the [Code of Conduct](CODE_OF_CONDUCT.md).

## Questions

Open a [discussion or issue](https://github.com/GSehurut/specinsure/issues). There are no silly questions, especially about the insurance concepts.
