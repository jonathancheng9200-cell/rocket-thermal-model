
import numpy as np

from coolant_flow import coolant_temperature_solver_separated
from design_constraints import evaluate_thermal_constraints


def run_mass_flow_sensitivity(
    x, dA_wall, T_aw, T_rad, h_g,
    mass_flow_values,
    T_c_in=300.0,
    c_p=3000.0,
    h_c=15000.0,
    L_w=0.003,
    k_w=350.0,
    epsilon_eff=0.20,
    T_wall_allow=1000.0,
    T_coolant_allow=600.0
):

    mass_flow_values = np.asarray(
        mass_flow_values, dtype=float
    )

    if (
        mass_flow_values.ndim != 1
        or mass_flow_values.size == 0
        or not np.all(np.isfinite(mass_flow_values))
        or np.any(mass_flow_values <= 0)
    ):
        raise ValueError(
            "Mass flow values must be a nonempty "
            "array of positive finite values."
        )

    results = []

    for m_dot_c in mass_flow_values:

        T_c, q, T_wg, T_wc, q_conv, q_rad = (
            coolant_temperature_solver_separated(
                x, dA_wall, T_aw, T_rad, h_g,
                T_c_in, m_dot_c, c_p,
                h_c, L_w, k_w, epsilon_eff,
                direction="counterflow"
            )
        )

        constraints = evaluate_thermal_constraints(
            T_wg,
            T_wc,
            T_c,
            T_wall_allow=T_wall_allow,
            T_coolant_allow=T_coolant_allow
        )

        Q_total = m_dot_c * c_p * (
            T_c[0] - T_c[-1]
        )

        results.append({
            "m_dot_c": float(m_dot_c),
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
