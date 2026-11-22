import math
from scipy.optimize import brentq

def thermal_resistance(T_g, T_c, h_g, h_c, L_w, k_w):


    # Validate physical parameters
    if h_g <= 0:
        raise ValueError("Gas-side heat transfer coefficient must be positive.")

    if h_c <= 0:
        raise ValueError("Coolant-side heat transfer coefficient must be positive.")

    if L_w <= 0:
        raise ValueError("Wall thickness must be positive.")

    if k_w <= 0:
        raise ValueError("Wall thermal conductivity must be positive.")

    parameters = [T_g, T_c, h_g, h_c, L_w, k_w]

    if not all(math.isfinite(value) for value in parameters):
        raise ValueError("All inputs must be finite numbers.")

    if T_g <= 0 or T_c <= 0:
        raise ValueError("Temperatures must be positive in Kelvin.")

    # Gas-side convective resistance
    R_g = 1 / h_g

    # Wall conduction resistance
    R_w = L_w / k_w

    # Coolant-side convective resistance
    R_c = 1 / h_c

    # Total thermal resistance per unit area
    R_total = R_g + R_w + R_c

    # Heat flux through the wall
    q_flux = (T_g - T_c) / R_total

    # Hot-gas-side wall temperature
    T_wg = T_g - q_flux / h_g

    # Coolant-side wall temperature
    T_wc = T_c + q_flux / h_c

    return q_flux, T_wg, T_wc


def thermal_resistance_radiation(
    T_g, T_c, h_g, h_c, L_w, k_w, epsilon_eff
):
    sigma_sb = 5.670374419e-8

    # Validate inputs
    parameters = [T_g, T_c, h_g, h_c, L_w, k_w, epsilon_eff]

    if not all(math.isfinite(value) for value in parameters):
        raise ValueError("All inputs must be finite.")

    if T_g <= 0 or T_c <= 0:
        raise ValueError("Temperatures must be positive in Kelvin.")

    if h_g <= 0 or h_c <= 0 or L_w <= 0 or k_w <= 0:
        raise ValueError("Thermal properties must be positive.")

    if not 0 <= epsilon_eff <= 1:
        raise ValueError("Effective radiation factor must be between 0 and 1.")

    # Resistance from hot-side wall to coolant
    R_downstream = L_w / k_w + 1 / h_c

    # Heat-balance residual
    def residual(T_wg):

        q_conv = h_g * (T_g - T_wg)

        q_rad = (
            epsilon_eff * sigma_sb
            * (T_g**4 - T_wg**4)
        )

        q_downstream = (T_wg - T_c) / R_downstream

        return q_conv + q_rad - q_downstream

    # Handle equal gas and coolant temperatures
    if T_g == T_c:
        T_wg = T_g
    else:
        # Solve for hot-side wall temperature
        T_wg = brentq(
            residual,
            min(T_g, T_c),
            max(T_g, T_c)
        )

    # Calculate total heat flux
    q_flux = (T_wg - T_c) / R_downstream

    # Coolant-side wall temperature
    T_wc = T_c + q_flux / h_c

    return q_flux, T_wg, T_wc


if __name__ == "__main__":

    # Baseline thermal parameters
    T_g = 3000.0
    T_c = 300.0

    h_g = 5000.0
    h_c = 15000.0

    L_w = 0.003
    k_w = 350.0

    # Run thermal solver
    q_flux, T_wg, T_wc = thermal_resistance(
        T_g, T_c, h_g, h_c, L_w, k_w
    )

    # Display results
    print(f"Heat flux: {q_flux / 1e6:.4f} MW/m²")
    print(f"Hot-side wall temperature: {T_wg:.2f} K")
    print(f"Coolant-side wall temperature: {T_wc:.2f} K")


    # Verify conduction through the wall
    q_conduction = k_w * (T_wg - T_wc) / L_w

    # Relative error between heat flux calculations
    relative_error = abs(q_flux - q_conduction) / abs(q_flux)

    print(f"Conduction heat flux: {q_conduction / 1e6:.4f} MW/m²")
    print(f"Relative error: {relative_error:.3e}")


    epsilon_eff = 0.20

    q_rad, T_wg_rad, T_wc_rad = thermal_resistance_radiation(
        T_g, T_c, h_g, h_c, L_w, k_w, epsilon_eff
    )

    print("\n--- Radiation Model ---")
    print(f"Heat flux: {q_rad / 1e6:.4f} MW/m²")
    print(f"Hot-side wall temperature: {T_wg_rad:.2f} K")
    print(f"Coolant-side wall temperature: {T_wc_rad:.2f} K")

    increase = (q_rad - q_flux) / q_flux * 100

    print(f"Heat flux increase: {increase:.2f}%")

