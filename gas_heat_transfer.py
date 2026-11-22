
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from nozzle_interface import get_nozzle_data


def gas_heat_transfer_coefficient(
    rho, u, radius,
    mu_g=1.0e-4,
    k_g=0.15,
    Pr=0.70
):

    rho = np.asarray(rho, dtype=float)
    u = np.asarray(u, dtype=float)
    radius = np.asarray(radius, dtype=float)

    if rho.shape != u.shape or rho.shape != radius.shape:
        raise ValueError("Gas property arrays must have matching shapes.")

    if not (
        np.all(np.isfinite(rho))
        and np.all(np.isfinite(u))
        and np.all(np.isfinite(radius))
    ):
        raise ValueError("Gas property arrays must be finite.")

    if np.any(rho <= 0) or np.any(radius <= 0):
        raise ValueError("Gas density and radius must be positive.")

    if np.any(u < 0):
        raise ValueError("Gas velocity cannot be negative.")

    if not np.all(np.isfinite([mu_g, k_g, Pr])):
        raise ValueError("Gas transport properties must be finite.")

    if mu_g <= 0 or k_g <= 0 or Pr <= 0:
        raise ValueError("Gas transport properties must be positive.")

    # Circular nozzle hydraulic diameter
    D_h = 2.0 * radius

    # Local Reynolds number
    Re = rho * u * D_h / mu_g

    # Simplified turbulent Nusselt correlation
    Nu = 0.023 * Re**0.8 * Pr**0.4

    # Gas-side heat-transfer coefficient
    h_g = Nu * k_g / D_h

    return h_g, Re, Nu

if __name__ == "__main__":

    data = get_nozzle_data()

    h_g, Re, Nu = gas_heat_transfer_coefficient(
        data["rho"],
        data["u"],
        data["r"]
    )

    throat_index = np.argmin(data["A"])
    peak_index = np.argmax(h_g)

    print("--- Gas-Side Heat Transfer ---")

    print(f"Inlet Reynolds number: {Re[0]:.3e}")
    print(f"Throat Reynolds number: {Re[throat_index]:.3e}")
    print(f"Exit Reynolds number: {Re[-1]:.3e}")

    print(f"Inlet heat-transfer coefficient: {h_g[0]:.2f} W/(m² K)")
    print(f"Throat heat-transfer coefficient: {h_g[throat_index]:.2f} W/(m² K)")
    print(f"Exit heat-transfer coefficient: {h_g[-1]:.2f} W/(m² K)")

    print(f"Maximum heat-transfer coefficient: {np.max(h_g):.2f} W/(m² K)")
    print(f"Peak heat-transfer location: {data['x'][peak_index]:.4f} m")

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))

    plt.plot(
        data["x"],
        h_g,
        linewidth=2,
        label="Gas-side heat-transfer coefficient"
    )

    plt.axvline(
        0.0,
        color="gray",
        linestyle="--",
        label="Throat"
    )

    plt.axvline(
        data["x"][peak_index],
        color="red",
        linestyle=":",
        label=f"Peak: x = {data['x'][peak_index]:.4f} m"
    )

    plt.xlabel("Axial position (m)")
    plt.ylabel("Heat-transfer coefficient (W/m² K)")
    plt.title("Gas-Side Heat Transfer Along the Nozzle")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        results_dir / "gas_heat_transfer_coefficient.png",
        dpi=300
    )

    plt.show()
