
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from project1.nozzle_simulator import (
    nozzle_flow,
    BASE_EXPANSION_RATIO,
    gamma,
    R,
    Tc
)


def get_nozzle_data(expansion_ratio=BASE_EXPANSION_RATIO):

    # Run Project 1 nozzle flow solver
    flow = nozzle_flow(expansion_ratio)

    # Extract nozzle geometry
    x = flow["x"]
    r = flow["r"]
    A = flow["A"]

    # Extract gas flow properties
    Mach = flow["Mach"]
    T = flow["T"]
    P = flow["P"]
    rho = flow["rho"]
    u = flow["u"]

    # Specific heat at constant pressure
    cp_gas = gamma * R / (gamma - 1)

    # Stagnation temperature consistency check
    T0_check = T + u**2 / (2 * cp_gas)

    return {
        "x": x,
        "r": r,
        "A": A,
        "Mach": Mach,
        "T": T,
        "P": P,
        "rho": rho,
        "u": u,
        "cp_gas": cp_gas,
        "T0_check": T0_check
    }


def adiabatic_wall_temperature(
    T, u, cp_gas, Pr=0.70,
    boundary_layer="turbulent"
):

    T = np.asarray(T, dtype=float)
    u = np.asarray(u, dtype=float)

    if T.shape != u.shape:
        raise ValueError("T and u must have matching shapes.")

    if not np.all(np.isfinite(T)) or not np.all(np.isfinite(u)):
        raise ValueError("Gas temperature and velocity must be finite.")

    if np.any(T <= 0):
        raise ValueError("Gas temperature must be positive.")

    if not np.isfinite(cp_gas) or cp_gas <= 0:
        raise ValueError("Gas specific heat must be positive and finite.")

    if not np.isfinite(Pr) or Pr <= 0:
        raise ValueError("Prandtl number must be positive and finite.")

    if boundary_layer == "turbulent":
        recovery_factor = Pr ** (1 / 3)

    elif boundary_layer == "laminar":
        recovery_factor = np.sqrt(Pr)

    else:
        raise ValueError(
            "Boundary layer must be 'turbulent' or 'laminar'."
        )

    T_aw = T + recovery_factor * u**2 / (2 * cp_gas)

    return T_aw, recovery_factor


if __name__ == "__main__":

    data = get_nozzle_data()

    throat_index = np.argmin(data["A"])

    print("--- Project 1 Nozzle Integration ---")
    print(f"Number of axial points: {len(data['x'])}")
    print(f"Nozzle inlet: {data['x'][0]:.4f} m")
    print(f"Throat location: {data['x'][throat_index]:.4f} m")
    print(f"Nozzle exit: {data['x'][-1]:.4f} m")

    print(f"Throat Mach number: {data['Mach'][throat_index]:.4f}")
    print(f"Exit Mach number: {data['Mach'][-1]:.4f}")

    print(f"Exit static temperature: {data['T'][-1]:.2f} K")
    print(f"Exit static pressure: {data['P'][-1] / 1e6:.4f} MPa")

    print(f"Gas specific heat: {data['cp_gas']:.2f} J/(kg K)")

    max_T0_error = np.max(np.abs(data["T0_check"] - Tc))

    print(f"Maximum stagnation temperature error: {max_T0_error:.3e} K")

    # Calculate adiabatic wall temperature
    T_aw, recovery_factor = adiabatic_wall_temperature(
        data["T"],
        data["u"],
        data["cp_gas"],
        Pr=0.70,
        boundary_layer="turbulent"
    )

    print("\n--- Adiabatic Wall Temperature ---")
    print(f"Recovery factor: {recovery_factor:.4f}")

    print(f"Inlet static temperature: {data['T'][0]:.2f} K")
    print(f"Inlet adiabatic wall temperature: {T_aw[0]:.2f} K")

    print(f"Exit static temperature: {data['T'][-1]:.2f} K")
    print(f"Exit adiabatic wall temperature: {T_aw[-1]:.2f} K")

    print(f"Maximum adiabatic wall temperature: {np.max(T_aw):.2f} K")

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))

    plt.plot(
        data["x"], data["T"],
        label="Static gas temperature",
        linewidth=2
    )

    plt.plot(
        data["x"], T_aw,
        label="Adiabatic wall temperature",
        linewidth=2
    )

    plt.plot(
        data["x"], data["T0_check"],
        linestyle="--",
        label="Stagnation temperature",
        linewidth=2
    )

    plt.axvline(
        0.0,
        color="gray",
        linestyle=":",
        label="Throat"
    )

    plt.xlabel("Axial position (m)")
    plt.ylabel("Temperature (K)")
    plt.title("Nozzle Gas Temperature Comparison")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        results_dir / "nozzle_temperature_comparison.png",
        dpi=300
    )

    plt.show()
