
import numpy as np
import pytest

from axial_thermal import axial_thermal_solver
from thermal_resistance import thermal_resistance_radiation


def test_output_dimensions():

    x = np.linspace(0, 0.5, 100)
    T_g = np.linspace(3000, 2000, 100)
    h_g = np.full(100, 5000.0)

    q, T_wg, T_wc = axial_thermal_solver(
        x, T_g, h_g,
        300, 15000, 0.003, 350, 0.20
    )

    assert q.shape == x.shape
    assert T_wg.shape == x.shape
    assert T_wc.shape == x.shape

def test_temperature_ordering():

    x = np.linspace(0, 0.5, 100)
    T_g = np.linspace(3000, 2000, 100)
    h_g = np.full(100, 5000.0)

    T_c = 300.0

    q, T_wg, T_wc = axial_thermal_solver(
        x, T_g, h_g,
        T_c, 15000, 0.003, 350, 0.20
    )

    assert np.all(T_g > T_wg)
    assert np.all(T_wg > T_wc)
    assert np.all(T_wc > T_c)
    assert np.all(q > 0)

def test_local_solver_consistency():

    x = np.array([0.0, 0.25, 0.5])
    T_g = np.array([3000.0, 2500.0, 2000.0])
    h_g = np.array([4000.0, 10000.0, 4000.0])

    q, T_wg, T_wc = axial_thermal_solver(
        x, T_g, h_g,
        300, 15000, 0.003, 350, 0.20
    )

    for i in range(len(x)):

        q_ref, T_wg_ref, T_wc_ref = (
            thermal_resistance_radiation(
                T_g[i], 300, h_g[i],
                15000, 0.003, 350, 0.20
            )
        )

        assert np.isclose(q[i], q_ref)
        assert np.isclose(T_wg[i], T_wg_ref)
        assert np.isclose(T_wc[i], T_wc_ref)

def test_spatial_convergence():

    peak_fluxes = []

    for N in [50, 100, 200, 400]:

        x = np.linspace(0, 0.5, N)

        T_g = 3000 - 1000 * (x / 0.5)

        h_g = 4000 + 6000 * np.exp(
            -((x - 0.25) / 0.08)**2
        )

        q, _, _ = axial_thermal_solver(
            x, T_g, h_g,
            300, 15000, 0.003, 350, 0.20
        )

        peak_fluxes.append(np.max(q))

    # Compare the two finest grids
    relative_difference = (
        abs(peak_fluxes[-1] - peak_fluxes[-2])
        / abs(peak_fluxes[-1])
    )

    assert relative_difference < 0.001
