
import numpy as np

from coolant_flow import coolant_temperature_solver_geometry
from nozzle_wall_geometry import nozzle_wall_segment_areas

def test_geometry_energy_balance():

    x = np.linspace(0, 0.5, 100)
    radius = np.linspace(0.10, 0.05, 100)

    dA, _ = nozzle_wall_segment_areas(x, radius)

    T_g = np.full_like(x, 2500.0)
    h_g = np.full_like(x, 5000.0)

    m_dot_c = 2.0
    c_p = 3000.0

    T_c, q, _, _ = coolant_temperature_solver_geometry(
        x, dA, T_g, h_g,
        300.0, m_dot_c, c_p,
        15000.0, 0.003, 350.0, 0.20,
        direction="counterflow"
    )

    Q_coolant = m_dot_c * c_p * (T_c[0] - T_c[-1])

    # Counterflow uses the right-hand node for each segment
    Q_wall = np.sum(q[1:] * dA)

    assert np.isclose(
        Q_coolant, Q_wall, rtol=1e-10
    )

def test_geometry_flow_directions():

    x = np.linspace(0, 0.5, 100)
    radius = np.full_like(x, 0.05)

    dA, _ = nozzle_wall_segment_areas(x, radius)

    T_g = np.full_like(x, 2500.0)
    h_g = np.full_like(x, 5000.0)

    args = (
        x, dA, T_g, h_g,
        300.0, 2.0, 3000.0,
        15000.0, 0.003, 350.0, 0.20
    )

    T_co, _, _, _ = coolant_temperature_solver_geometry(
        *args, direction="coflow"
    )

    T_counter, _, _, _ = coolant_temperature_solver_geometry(
        *args, direction="counterflow"
    )

    assert np.isclose(T_co[0], 300.0)
    assert np.isclose(T_counter[-1], 300.0)

    assert T_co[-1] > T_co[0]
    assert T_counter[0] > T_counter[-1]

    # Symmetric constant-property problem:
    # reversing flow reverses the temperature distribution
    assert np.allclose(T_co, T_counter[::-1])

import pytest


def test_geometry_invalid_area():

    x = np.linspace(0, 0.5, 10)

    dA = np.full(9, 0.001)
    dA[4] = -0.001

    T_g = np.full(10, 2500.0)
    h_g = np.full(10, 5000.0)

    with pytest.raises(ValueError):
        coolant_temperature_solver_geometry(
            x, dA, T_g, h_g,
            300.0, 2.0, 3000.0,
            15000.0, 0.003, 350.0, 0.20,
            direction="counterflow"
        )
