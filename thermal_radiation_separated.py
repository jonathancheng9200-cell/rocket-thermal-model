
import numpy as np
from scipy.optimize import brentq

SIGMA = 5.670374419e-8


def thermal_resistance_separated(
    T_aw,
    T_rad,
    T_c,
    h_g,
    h_c,
    L_w,
    k_w,
    epsilon_eff
):
    values = [
        T_aw, T_rad, T_c,
        h_g, h_c, L_w, k_w, epsilon_eff
    ]

    if not np.all(np.isfinite(values)):
        raise ValueError("All inputs must be finite.")

    if T_aw <= 0 or T_rad <= 0 or T_c <= 0:
        raise ValueError("Temperatures must be positive.")

    if h_g <= 0 or h_c <= 0 or L_w <= 0 or k_w <= 0:
        raise ValueError("Thermal properties must be positive.")

    if not 0 <= epsilon_eff <= 1:
        raise ValueError("Emissivity must be between 0 and 1.")

    # Wall conduction + coolant-side convection resistance
    R_out = L_w / k_w + 1.0 / h_c

    def energy_residual(T_wg):
        q_conv = h_g * (T_aw - T_wg)

        q_rad = epsilon_eff * SIGMA * (
            T_rad**4 - T_wg**4
        )

        q_wall = (T_wg - T_c) / R_out

        return q_conv + q_rad - q_wall

    # Bracket the physical solution for positive temperatures
    lower = min(T_c, T_aw, T_rad)
    upper = max(T_c, T_aw, T_rad)

    T_wg = brentq(energy_residual, lower, upper)

    q_conv = h_g * (T_aw - T_wg)

    q_rad = epsilon_eff * SIGMA * (
        T_rad**4 - T_wg**4
    )

    q_total = q_conv + q_rad

    T_wc = T_c + q_total / h_c

    return q_total, T_wg, T_wc, q_conv, q_rad

if __name__ == "__main__":

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

    print("--- Separated Convection and Radiation ---")

    print(f"Total heat flux: {q / 1e6:.4f} MW/m²")
    print(f"Convective heat flux: {q_conv / 1e6:.4f} MW/m²")
    print(f"Radiative heat flux: {q_rad / 1e6:.4f} MW/m²")

    print(f"Hot-side wall temperature: {T_wg:.2f} K")
    print(f"Coolant-side wall temperature: {T_wc:.2f} K")

    residual = abs(q - (T_wg - T_wc) * 350.0 / 0.003)

    print(f"Conduction balance residual: {residual:.3e} W/m²")
