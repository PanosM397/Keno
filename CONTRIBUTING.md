# Contributing to Keno

Thanks for your interest in Keno. This is an independent research software
prototype for template-free gravitational-wave residual search after generative
noise subtraction.

## Ways to contribute

- Open an issue for bugs, unclear docs, or reproducibility problems
- Suggest improvements to install/run paths, tests, or evaluation tooling
- Small, focused pull requests are welcome

Please keep scientific claims aligned with
[`docs/paper/SUBMISSION.md`](docs/paper/SUBMISSION.md) (claim audit): Keno is
complementary to BBH classifiers; catalog follow-up is a consistency check, not
discovery; do not overclaim residual-vs-raw injection results.

## Development setup

See the root [`README.md`](README.md) for the three-service quickstart
(ML engine, Express backend, Angular UI).

ML engine:

```bash
cd ml-engine
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest scipy
pytest
```

Backend:

```bash
cd backend
npm install
node --test
```

## Pull requests

1. Fork and create a branch from `main`
2. Keep changes focused
3. Add or update tests when changing evaluation metrics or cache/API helpers
4. Describe what changed and how you checked it

## Support expectations

This is a solo-maintained research prototype. Best-effort responses to issues
are the goal; there is no SLA. Security-sensitive reports can be emailed to
`panosmng97@gmail.com`.

## Code of conduct

Be respectful and constructive. Harassment or bad-faith interaction is not
tolerated.
