"""Exercise the native CBC adapter with real continuous and integer programs."""

import os

import pulp
import pytest

from ntqr_allotment.solver import configure_cbc, runtime_provenance


def _configure_real_cbc() -> None:
    try:
        configure_cbc()
    except RuntimeError:
        if os.environ.get("NTQR_CBC_PATH") is not None:
            raise
        pytest.skip("real CBC solver unavailable")


@pytest.mark.parametrize("category, expected", [("Continuous", 1.5), ("Integer", 1.0)])
def test_native_cbc_solves_real_program(category, expected):
    pulp.PULP_CBC_CMD._permissions_ok = False
    _configure_real_cbc()
    problem = pulp.LpProblem("native_cbc_smoke", pulp.LpMaximize)
    x = pulp.LpVariable("x", lowBound=0, cat=category)
    problem += x
    problem += 2 * x <= 3
    status = problem.solve(pulp.PULP_CBC_CMD(msg=False))
    assert pulp.LpStatus[status] == "Optimal"
    assert x.value() == pytest.approx(expected)


def test_runtime_provenance_has_versions_without_host_paths():
    metadata = runtime_provenance()
    assert metadata["ntqr_version"] == "0.9"
    assert metadata["cbc_version"] != "unknown"
    assert all("/" not in value for value in metadata.values())
