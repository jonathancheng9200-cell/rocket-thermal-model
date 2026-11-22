
import math

from thermal_resistance import (
    thermal_resistance,
    thermal_resistance_radiation
)


def test_baseline():

    q, T_wg, T_wc = thermal_resistance(
        T_g=3000,
        T_c=300,
        h_g=5000,
        h_c=15000,
        L_w=0.003,
        k_w=350
    )

    assert math.isclose(q / 1e6, 9.8097, rel_tol=1e-4)
    assert math.isclose(T_wg, 1038.06, rel_tol=1e-4)
    assert math.isclose(T_wc, 953.98, rel_tol=1e-4)


def test_energy_consistency():

    k_w = 350
    L_w = 0.003

    q, T_wg, T_wc = thermal_resistance(
        3000, 300, 5000, 15000, L_w, k_w
    )

    q_conduction = k_w * (T_wg - T_wc) / L_w

    assert math.isclose(q, q_conduction, rel_tol=1e-10)


def test_invalid_conductivity():

    try:
        thermal_resistance(
            3000, 300, 5000, 15000, 0.003, -350
        )

    except ValueError:
        return

    raise AssertionError(
        "Expected ValueError for negative conductivity."
    )


def test_radiation_baseline():

    q, T_wg, T_wc = thermal_resistance_radiation(
        T_g=3000,
        T_c=300,
        h_g=5000,
        h_c=15000,
        L_w=0.003,
        k_w=350,
        epsilon_eff=0.20
    )

    assert math.isclose(q / 1e6, 10.4657, rel_tol=1e-4)
    assert math.isclose(T_wg, 1087.42, rel_tol=1e-4)
    assert math.isclose(T_wc, 997.71, rel_tol=1e-4)


def test_radiation_energy_balance():

    T_g = 3000.0
    T_c = 300.0
    h_g = 5000.0
    h_c = 15000.0
    L_w = 0.003
    k_w = 350.0
    epsilon_eff = 0.20

    sigma_sb = 5.670374419e-8

    q, T_wg, T_wc = thermal_resistance_radiation(
        T_g, T_c, h_g, h_c, L_w, k_w, epsilon_eff
    )

    # Gas-side convection
    q_conv = h_g * (T_g - T_wg)

    # Gas-side radiation
    q_rad = (
        epsilon_eff * sigma_sb
        * (T_g**4 - T_wg**4)
    )

    # Conduction through wall
    q_cond = k_w * (T_wg - T_wc) / L_w

    # Heat transfer into coolant
    q_coolant = h_c * (T_wc - T_c)

    assert math.isclose(
        q_conv + q_rad, q, rel_tol=1e-10
    )

    assert math.isclose(
        q_cond, q, rel_tol=1e-10
    )

    assert math.isclose(
        q_coolant, q, rel_tol=1e-10
    )


def test_zero_radiation_limit():

    q_old, T_wg_old, T_wc_old = thermal_resistance(
        3000, 300, 5000, 15000, 0.003, 350
    )

    q_new, T_wg_new, T_wc_new = thermal_resistance_radiation(
        3000, 300, 5000, 15000, 0.003, 350, 0.0
    )

    assert math.isclose(q_old, q_new, rel_tol=1e-10)
    assert math.isclose(T_wg_old, T_wg_new, rel_tol=1e-10)
    assert math.isclose(T_wc_old, T_wc_new, rel_tol=1e-10)
