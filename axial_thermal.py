
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from thermal_resistance import thermal_resistance_radiation


def axial_thermal_solver(
    x, T_g, h_g, T_c, h_c, L_w, k_w, epsilon_eff
):

    x = np.asarray(x, dtype=float)
    T_g = np.asarray(T_g, dtype=float)
    h_g = np.asarray(h_g, dtype=float)

    # Check array dimensions
    if x.ndim != 1 or T_g.ndim != 1 or h_g.ndim != 1:
        raise ValueError("Inputs x, T_g, and h_g must be 1D arrays.")

    if not (len(x) == len(T_g) == len(h_g)):
        raise ValueError("Axial arrays must have equal lengths.")

    # Initialize output arrays
    q_flux = np.zeros_like(x)
    T_wg = np.zeros_like(x)
    T_wc = np.zeros_like(x)

    # Solve thermal balance at each axial location
    for i in range(len(x)):

        q_flux[i], T_wg[i], T_wc[i] = (
            thermal_resistance_radiation(
                T_g[i],
                T_c,
                h_g[i],
                h_c,
                L_w,
                k_w,
                epsilon_eff
            )
        )

    return q_flux, T_wg, T_wc


if __name__ == "__main__":

    # Axial locations (m)
    x = np.linspace(0, 0.5, 100)

    # Illustrative gas temperature distribution (K)
    T_g = 3000 - 1000 * (x / 0.5)

    # Illustrative gas-side heat transfer coefficient
    h_g = 4000 + 6000 * np.exp(
        -((x - 0.25) / 0.08)**2
    )

    # Constant coolant and wall properties
    T_c = 300.0
    h_c = 15000.0
    L_w = 0.003
    k_w = 350.0
    epsilon_eff = 0.20

    q_flux, T_wg, T_wc = axial_thermal_solver(
        x, T_g, h_g, T_c, h_c, L_w, k_w, epsilon_eff
    )

    print(f"Maximum heat flux: {np.max(q_flux) / 1e6:.4f} MW/m²")
    print(f"Maximum hot-side wall temperature: {np.max(T_wg):.2f} K")
    print(f"Maximum coolant-side wall temperature: {np.max(T_wc):.2f} K")

    max_index = np.argmax(q_flux)

    print(f"Peak heat flux location: {x[max_index]:.4f} m")

    max_Twg_index = np.argmax(T_wg)
    max_Twc_index = np.argmax(T_wc)

    print(
        f"Peak hot-side wall temperature location: "
        f"{x[max_Twg_index]:.4f} m"
    )

    print(
        f"Peak coolant-side wall temperature location: "
        f"{x[max_Twc_index]:.4f} m"
    )

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Plot axial heat flux
    plt.figure(figsize=(9, 5))

    plt.plot(
        x,
        q_flux / 1e6,
        linewidth=2,
        label="Wall heat flux"
    )

    plt.axvline(
        x[max_index],
        color="red",
        linestyle="--",
        label=f"Peak: x = {x[max_index]:.3f} m"
    )

    plt.xlabel("Axial position (m)")
    plt.ylabel("Heat flux (MW/m²)")
    plt.title("Axial Heat Flux Distribution")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "axial_heat_flux.png", dpi=300)
    plt.show()

    # Plot gas and wall temperatures
    plt.figure(figsize=(9, 5))

    plt.plot(x, T_g, label="Gas temperature", linewidth=2)

    plt.plot(
        x, T_wg,
        label="Hot-side wall temperature",
        linewidth=2
    )

    plt.plot(
        x, T_wc,
        label="Coolant-side wall temperature",
        linewidth=2
    )

    plt.axhline(
        T_c,
        color="gray",
        linestyle="--",
        label="Coolant bulk temperature"
    )

    plt.xlabel("Axial position (m)")
    plt.ylabel("Temperature (K)")
    plt.title("Axial Thermal Distribution")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(results_dir / "axial_temperatures.png", dpi=300)
    plt.show()

