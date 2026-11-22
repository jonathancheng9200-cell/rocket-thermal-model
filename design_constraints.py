
import numpy as np


def evaluate_thermal_constraints(
    T_wg,
    T_wc,
    T_c,
    T_wall_allow=1000.0,
    T_coolant_allow=600.0
):
    T_wg = np.asarray(T_wg, dtype=float)
    T_wc = np.asarray(T_wc, dtype=float)
    T_c = np.asarray(T_c, dtype=float)

    if (
        T_wg.ndim != 1
        or T_wc.shape != T_wg.shape
        or T_c.shape != T_wg.shape
        or T_wg.size == 0
    ):
        raise ValueError("Temperature arrays must be nonempty and aligned.")

    if not all(
        np.all(np.isfinite(arr))
        for arr in (T_wg, T_wc, T_c)
    ):
        raise ValueError("Temperatures must be finite.")

    if (
        not np.isfinite(T_wall_allow)
        or not np.isfinite(T_coolant_allow)
        or T_wall_allow <= 0
        or T_coolant_allow <= 0
    ):
        raise ValueError("Temperature limits must be positive and finite.")

    max_hot_wall = np.max(T_wg)
    max_cold_wall = np.max(T_wc)
    max_coolant = np.max(T_c)

    wall_margin = T_wall_allow - max_hot_wall
    coolant_margin = T_coolant_allow - max_coolant

    return {
        "max_hot_wall": max_hot_wall,
        "max_cold_wall": max_cold_wall,
        "max_coolant": max_coolant,
        "wall_margin": wall_margin,
        "coolant_margin": coolant_margin,
        "wall_pass": wall_margin >= 0,
        "coolant_pass": coolant_margin >= 0
    }
