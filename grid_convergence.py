
import numpy as np

from nozzle_wall_geometry import nozzle_wall_segment_areas
from coolant_flow import coolant_temperature_solver_separated


def refine_grid(x, radius, T_aw, T_rad, h_g, factor):
    if not isinstance(factor, (int, np.integer)) or factor < 1:
        raise ValueError("Refinement factor must be a positive integer.")

    x = np.asarray(x, dtype=float)
    radius = np.asarray(radius, dtype=float)
    T_aw = np.asarray(T_aw, dtype=float)
    T_rad = np.asarray(T_rad, dtype=float)
    h_g = np.asarray(h_g, dtype=float)

    if x.ndim != 1 or len(x) < 2:
        raise ValueError("Invalid axial grid.")

    if any(arr.shape != x.shape for arr in
           (radius, T_aw, T_rad, h_g)):
        raise ValueError("Input arrays must have matching shapes.")

    if not all(np.all(np.isfinite(arr)) for arr in
               (x, radius, T_aw, T_rad, h_g)):
        raise ValueError("Inputs must be finite.")

    if not np.all(np.diff(x) > 0):
        raise ValueError("Axial positions must increase.")

    # Subdivide each original segment.
    # This preserves all original nodes, including the throat.
    fractions = np.arange(factor) / factor

    x_new = np.concatenate([
        (x[:-1, None] +
         np.diff(x)[:, None] * fractions).ravel(),
        x[-1:]
    ])

    radius_new = np.interp(x_new, x, radius)
    T_aw_new = np.interp(x_new, x, T_aw)
    T_rad_new = np.interp(x_new, x, T_rad)
    h_g_new = np.interp(x_new, x, h_g)

    return x_new, radius_new, T_aw_new, T_rad_new, h_g_new


def run_grid_convergence(
    x, radius, T_aw, T_rad, h_g,
    factors=(1, 2, 4, 8),
    T_c_in=300.0,
    m_dot_c=2.0,
    c_p=3000.0,
    h_c=15000.0,
    L_w=0.003,
    k_w=350.0,
    epsilon_eff=0.20
):
    results = []

    for factor in factors:
        xr, rr, Taw, Trad, hg = refine_grid(
            x, radius, T_aw, T_rad, h_g, factor
        )

        dA, _ = nozzle_wall_segment_areas(xr, rr)

        Tc, q, Twg, Twc, qc, qr = (
            coolant_temperature_solver_separated(
                xr, dA, Taw, Trad, hg,
                T_c_in, m_dot_c, c_p,
                h_c, L_w, k_w, epsilon_eff,
                direction="counterflow"
            )
        )

        results.append({
            "factor": factor,
            "points": len(xr),
            "T_c_out": float(Tc[0]),
            "T_wg_max": float(np.max(Twg)),
            "Q_total": float(
                m_dot_c * c_p * (Tc[0] - Tc[-1])
            )
        })

    return results
