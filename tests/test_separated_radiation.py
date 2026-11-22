
import numpy as np

from thermal_radiation_separated import (
    thermal_resistance_separated
)

from coolant_flow import (
    coolant_temperature_solver_separated
)

from nozzle_wall_geometry import (
    nozzle_wall_segment_areas
)


def test_local_radiation_energy_balance():

    q, T_wg, T_wc, q_conv, q_rad = (
        thermal_resistance_separated(
            T_aw=2900.0,
            T_rad=2500.0,
            T_c=350.0,
            h_g=4500.0,
            h_c=15000.0,
            L_w=0.003,
            k_w=350.0,
            epsilon_eff=0.20
        )
    )

    q_cond = (T_wg - T_wc) * 350.0 / 0.003

    assert np.isclose(q, q_conv + q_rad, rtol=1e-10)

    assert np.isclose(q, q_cond, rtol=1e-10)


def test_axial_radiation_energy_balance():

    x = np.linspace(0, 0.5, 100)
    radius = np.linspace(0.10, 0.05, 100)

    dA, _ = nozzle_wall_segment_areas(x, radius)

    T_aw = np.full_like(x, 2900.0)
    T_rad = np.full_like(x, 2500.0)
    h_g = np.full_like(x, 4500.0)

    m_dot_c = 2.0
    c_p = 3000.0

    T_c, q, _, _, q_conv, q_rad = (
        coolant_temperature_solver_separated(
            x, dA, T_aw, T_rad, h_g,
            300.0, m_dot_c, c_p,
            15000.0, 0.003, 350.0, 0.20,
            direction="counterflow"
        )
    )

    Q_coolant = m_dot_c * c_p * (
        T_c[0] - T_c[-1]
    )

    Q_wall = np.sum(q[1:] * dA)

    Q_components = np.sum(
        (q_conv[1:] + q_rad[1:]) * dA
    )

    assert np.isclose(
        Q_coolant, Q_wall, rtol=1e-10
    )

    assert np.isclose(
        Q_wall, Q_components, rtol=1e-10
    )


def test_local_and_axial_solver_consistency():

    x = np.array([0.0, 0.1])
    radius = np.array([0.05, 0.05])

    dA, _ = nozzle_wall_segment_areas(x, radius)

    T_aw = np.array([2900.0, 2900.0])
    T_rad = np.array([2500.0, 2500.0])
    h_g = np.array([4500.0, 4500.0])

    T_c, q, T_wg, T_wc, q_conv, q_rad = (
        coolant_temperature_solver_separated(
            x, dA, T_aw, T_rad, h_g,
            350.0, 2.0, 3000.0,
            15000.0, 0.003, 350.0, 0.20,
            direction="counterflow"
        )
    )

    local = thermal_resistance_separated(
        T_aw=2900.0,
        T_rad=2500.0,
        T_c=350.0,
        h_g=4500.0,
        h_c=15000.0,
        L_w=0.003,
        k_w=350.0,
        epsilon_eff=0.20
    )

    # The counterflow inlet is the last grid point
    assert np.isclose(q[-1], local[0])
    assert np.isclose(T_wg[-1], local[1])
    assert np.isclose(T_wc[-1], local[2])
    assert np.isclose(q_conv[-1], local[3])
    assert np.isclose(q_rad[-1], local[4])
