
import numpy as np
import pytest

from coolant_flow import (
    coolant_temperature_solver,
    coolant_pressure_solver,
    rectangular_channel_geometry,
    coolant_temperature_solver_directional,
    coolant_pressure_solver_directional
)

def test_coolant_energy_balance():

    x = np.linspace(0, 0.5, 100)
    T_g = 3000 - 1000 * (x / 0.5)
    h_g = 4000 + 6000 * np.exp(
        -((x - 0.25) / 0.08)**2
    )

    m_dot_c = 2.0
    c_p = 3000.0
    P_h = 0.10

    T_c, q, _, _ = coolant_temperature_solver(
        x, T_g, h_g,
        300, m_dot_c, c_p,
        P_h, 15000, 0.003, 350, 0.20
    )

    # Energy gained by coolant
    Q_coolant = m_dot_c * c_p * (T_c[-1] - T_c[0])

    # Energy transferred from wall
    ds = np.diff(x)
    Q_wall = np.sum(q[:-1] * P_h * ds)

    assert np.isclose(
        Q_coolant, Q_wall, rtol=1e-10
    )

def test_pressure_loss():

    x = np.linspace(0, 0.5, 100)

    P = coolant_pressure_solver(
        x,
        P_in=5e6,
        rho_c=1000,
        V_c=4,
        D_h=0.005,
        f_D=0.020
    )

    assert np.isclose(P[0], 5e6)
    assert np.isclose(P[-1], 4.984e6)

    assert np.all(np.diff(P) < 0)

def test_channel_geometry():

    A_total, D_h, V_c = rectangular_channel_geometry(
        n_channels=50,
        width=0.005,
        height=0.002,
        m_dot_c=2.0,
        rho_c=1000.0
    )

    assert np.isclose(A_total, 0.0005)
    assert np.isclose(D_h, 0.002857142857)
    assert np.isclose(V_c, 4.0)

    # Verify continuity
    assert np.isclose(
        1000.0 * V_c * A_total, 2.0
    )

def test_pressure_direction():

    x = np.linspace(0, 0.5, 100)

    P_co = coolant_pressure_solver_directional(
        x, 5e6, 1000, 4, 0.002857142857, 0.020,
        direction="coflow"
    )

    P_counter = coolant_pressure_solver_directional(
        x, 5e6, 1000, 4, 0.002857142857, 0.020,
        direction="counterflow"
    )

    assert np.isclose(P_co[0], 5e6)
    assert np.isclose(P_counter[-1], 5e6)

    assert np.isclose(P_co[-1], P_counter[0])

    assert np.all(np.diff(P_co) < 0)
    assert np.all(np.diff(P_counter) > 0)

def test_counterflow_temperature():

    x = np.linspace(0, 0.5, 100)

    T_g = 3000 - 1000 * (x / 0.5)

    h_g = 4000 + 6000 * np.exp(
        -((x - 0.25) / 0.08)**2
    )

    T_c, q, T_wg, T_wc = (
        coolant_temperature_solver_directional(
            x, T_g, h_g,
            300, 2.0, 3000,
            0.10, 15000, 0.003, 350, 0.20,
            direction="counterflow"
        )
    )

    # Coolant inlet is at nozzle exit
    assert np.isclose(T_c[-1], 300.0)

    # Coolant outlet is near chamber
    assert T_c[0] > T_c[-1]

    # Temperature decreases with increasing x
    assert np.all(np.diff(T_c) < 0)

    # Heat flows from gas toward coolant
    assert np.all(q > 0)
    assert np.all(T_wg > T_wc)
