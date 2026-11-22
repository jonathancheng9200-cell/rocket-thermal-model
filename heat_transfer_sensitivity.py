
import numpy as np

from coolant_flow import coolant_temperature_solver_separated
from design_constraints import evaluate_thermal_constraints


def run_hc_sensitivity(
    x, dA_wall, T_aw, T_rad, h_g,
    hc_values,
    T_c_in=300.0,
    m_dot_c=2.0,
    c_p=3000.0,
    L_w=0.003,
    k_w=350.0,
    epsilon_eff=0.20,
    T_wall_allow=1000.0,
    T_coolant_allow=600.0
):

    hc_values = np.asarray(hc_values, dtype=float)

    if (
        hc_values.ndim != 1
        or hc_values.size == 0
        or not np.all(np.isfinite(hc_values))
        or np.any(hc_values <= 0)
    ):
        raise ValueError(
            "Heat-transfer coefficients must be positive and finite."
        )

    results = []

    for h_c in hc_values:

        T_c, q, T_wg, T_wc, q_conv, q_rad = (
            coolant_temperature_solver_separated(
                x, dA_wall, T_aw, T_rad, h_g,
                T_c_in, m_dot_c, c_p,
                h_c, L_w, k_w, epsilon_eff,
                direction="counterflow"
            )
        )

        constraints = evaluate_thermal_constraints(
            T_wg, T_wc, T_c,
            T_wall_allow=T_wall_allow,
            T_coolant_allow=T_coolant_allow
        )

        Q_total = m_dot_c * c_p * (
            T_c[0] - T_c[-1]
        )

        results.append({
            "h_c": float(h_c),
            "T_c_out": float(T_c[0]),
            "T_wg_max": float(np.max(T_wg)),
            "T_wc_max": float(np.max(T_wc)),
            "Q_total": float(Q_total),
            "wall_margin": float(constraints["wall_margin"]),
            "coolant_margin": float(constraints["coolant_margin"]),
            "wall_pass": bool(constraints["wall_pass"]),
            "coolant_pass": bool(constraints["coolant_pass"])
        })

    return results

def run_kw_sensitivity(
    x, dA_wall, T_aw, T_rad, h_g,
    kw_values,
    T_c_in=300.0,
    m_dot_c=2.0,
    c_p=3000.0,
    h_c=15000.0,
    L_w=0.003,
    epsilon_eff=0.20,
    T_wall_allow=1000.0,
    T_coolant_allow=600.0
):

    kw_values = np.asarray(kw_values, dtype=float)

    if (
        kw_values.ndim != 1
        or kw_values.size == 0
        or not np.all(np.isfinite(kw_values))
        or np.any(kw_values <= 0)
    ):
        raise ValueError(
            "Conductivity values must be positive and finite."
        )

    results = []

    for k_w in kw_values:

        T_c, q, T_wg, T_wc, q_conv, q_rad = (
            coolant_temperature_solver_separated(
                x, dA_wall, T_aw, T_rad, h_g,
                T_c_in, m_dot_c, c_p,
                h_c, L_w, k_w, epsilon_eff,
                direction="counterflow"
            )
        )

        constraints = evaluate_thermal_constraints(
            T_wg, T_wc, T_c,
            T_wall_allow=T_wall_allow,
            T_coolant_allow=T_coolant_allow
        )

        Q_total = m_dot_c * c_p * (
            T_c[0] - T_c[-1]
        )

        results.append({
            "k_w": float(k_w),
            "T_c_out": float(T_c[0]),
            "T_wg_max": float(np.max(T_wg)),
            "T_wc_max": float(np.max(T_wc)),
            "Q_total": float(Q_total),
            "wall_margin": float(constraints["wall_margin"]),
            "coolant_margin": float(constraints["coolant_margin"]),
            "wall_pass": bool(constraints["wall_pass"]),
            "coolant_pass": bool(constraints["coolant_pass"])
        })

    return results
