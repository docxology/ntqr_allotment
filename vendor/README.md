# vendor/ — ntqr_allotment

- `ntqr-0.8-py2.py3-none-any.whl` — the published `ntqr` wheel, vendored because its sdist is broken (flit build references a README absent from the tarball, so `uv` cannot lock it from the index). Referenced via `[tool.uv.sources]` in `pyproject.toml`.
