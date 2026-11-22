
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from nozzle_interface import (
    get_nozzle_data,
    adiabatic_wall_temperature
)

from gas_heat_transfer import gas_heat_transfer_coefficient

from coolant_flow import (
    coolant_temperature_solver_directional,
    coolant_pressure_solver_directional,
    rectangular_channel_geometry,
    coolant_temperature_solver_geometry,
    coolant_temperature_solver_separated
)

from nozzle_wall_geometry import nozzle_wall_segment_areas

from design_constraints import evaluate_thermal_constraints

from mass_flow_sensitivity import run_mass_flow_sensitivity

from heat_transfer_sensitivity import (
    run_hc_sensitivity,
    run_kw_sensitivity
)

from grid_convergence import run_grid_convergence

from validation_report import generate_validation_report

if __name__ == "__main__":

    # Load Project 1 nozzle simulation
    data = get_nozzle_data()

    x = data["x"]

    # Calculate recovery temperature
    T_aw, recovery_factor = adiabatic_wall_temperature(
        data["T"],
        data["u"],
        data["cp_gas"],
        Pr=0.70,
        boundary_layer="turbulent"
    )

    # Calculate gas-side heat-transfer coefficient
    h_g, Re, Nu = gas_heat_transfer_coefficient(
        data["rho"],
        data["u"],
        data["r"]
    )

    # Coolant properties
    T_c_in = 300.0
    m_dot_c = 2.0
    c_p = 3000.0

    # Effective heated perimeter
    P_h = 0.10

    # Wall and heat-transfer properties
    h_c = 15000.0
    L_w = 0.003
    k_w = 350.0
    epsilon_eff = 0.20

    # Cooling channel geometry
    n_channels = 50
    width = 0.005
    height = 0.002

    # Coolant hydraulic properties
    P_in = 5.0e6
    rho_c = 1000.0
    f_D = 0.020

    A_total, D_h, V_c = rectangular_channel_geometry(
        n_channels,
        width,
        height,
        m_dot_c,
        rho_c
    )

    # Counterflow regenerative cooling
    T_c, q_flux, T_wg, T_wc = (
        coolant_temperature_solver_directional(
            x,
            T_aw,
            h_g,
            T_c_in,
            m_dot_c,
            c_p,
            P_h,
            h_c,
            L_w,
            k_w,
            epsilon_eff,
            direction="counterflow"
        )
    )

    # Counterflow coolant pressure
    P_c = coolant_pressure_solver_directional(
        x,
        P_in,
        rho_c,
        V_c,
        D_h,
        f_D,
        direction="counterflow"
    )

    throat_index = np.argmin(data["A"])
    peak_flux_index = np.argmax(q_flux)
    peak_wall_index = np.argmax(T_wg)

    Q_total = m_dot_c * c_p * (T_c[0] - T_c[-1])
    delta_P = P_c[-1] - P_c[0]

    print("--- Integrated Nozzle Thermal Model ---")

    print(f"Nozzle length: {x[-1] - x[0]:.4f} m")
    print(f"Throat location: {x[throat_index]:.4f} m")

    print(f"Coolant inlet temperature: {T_c[-1]:.2f} K")
    print(f"Coolant outlet temperature: {T_c[0]:.2f} K")

    print(f"Total absorbed heat: {Q_total / 1e3:.2f} kW")

    print(f"Coolant inlet pressure: {P_c[-1] / 1e6:.4f} MPa")
    print(f"Coolant outlet pressure: {P_c[0] / 1e6:.4f} MPa")
    print(f"Coolant pressure loss: {delta_P / 1e3:.2f} kPa")

    print(f"Maximum heat flux: {np.max(q_flux) / 1e6:.4f} MW/m²")
    print(f"Peak heat flux location: {x[peak_flux_index]:.4f} m")

    print(f"Maximum hot-side wall temperature: {np.max(T_wg):.2f} K")
    print(f"Peak wall temperature location: {x[peak_wall_index]:.4f} m")

    # Calculate actual nozzle wall surface areas
    dA_wall, ds_wall = nozzle_wall_segment_areas(
        x,
        data["r"]
    )

    # Geometry-based counterflow cooling
    T_c_geom, q_geom, T_wg_geom, T_wc_geom = (
        coolant_temperature_solver_geometry(
            x,
            dA_wall,
            T_aw,
            h_g,
            T_c_in,
            m_dot_c,
            c_p,
            h_c,
            L_w,
            k_w,
            epsilon_eff,
            direction="counterflow"
        )
    )

    # Calculate total absorbed heat
    Q_geom = m_dot_c * c_p * (
        T_c_geom[0] - T_c_geom[-1]
    )

    print("\n--- Geometry-Based Cooling Results ---")

    print(f"Total wall area: {np.sum(dA_wall):.6f} m²")

    print(
        f"Coolant outlet temperature: "
        f"{T_c_geom[0]:.2f} K"
    )

    print(
        f"Total absorbed heat: "
        f"{Q_geom / 1e3:.2f} kW"
    )

    print(
        f"Maximum wall heat flux: "
        f"{np.max(q_geom) / 1e6:.4f} MW/m²"
    )

    print(
        f"Maximum hot-side wall temperature: "
        f"{np.max(T_wg_geom):.2f} K"
    )

    print("\n--- Old vs Geometry-Based Model ---")

    print(
        f"Old coolant outlet: {T_c[0]:.2f} K"
    )

    print(
        f"Geometry-based outlet: {T_c_geom[0]:.2f} K"
    )

    print(
        f"Old absorbed heat: {Q_total / 1e3:.2f} kW"
    )

    print(
        f"Geometry-based absorbed heat: {Q_geom / 1e3:.2f} kW"
    )

    # Separate gas-side convection and radiation
    T_c_sep, q_sep, T_wg_sep, T_wc_sep, q_conv, q_rad = (
        coolant_temperature_solver_separated(
            x,
            dA_wall,
            T_aw,
            data["T"],
            h_g,
            T_c_in,
            m_dot_c,
            c_p,
            h_c,
            L_w,
            k_w,
            epsilon_eff,
            direction="counterflow"
        )
    )

    Q_sep = m_dot_c * c_p * (
        T_c_sep[0] - T_c_sep[-1]
    )

    # Use the upstream node in coolant-flow direction
    Q_conv = np.sum(q_conv[1:] * dA_wall)
    Q_rad = np.sum(q_rad[1:] * dA_wall)

    print("\n--- Separated Radiation Cooling Results ---")

    print(f"Coolant outlet temperature: {T_c_sep[0]:.2f} K")
    print(f"Total absorbed heat: {Q_sep / 1e3:.2f} kW")

    print(f"Convective heat transfer: {Q_conv / 1e3:.2f} kW")
    print(f"Radiative heat transfer: {Q_rad / 1e3:.2f} kW")

    print(f"Maximum total heat flux: {np.max(q_sep) / 1e6:.4f} MW/m²")
    print(f"Maximum hot-side wall temperature: {np.max(T_wg_sep):.2f} K")

    print(
        f"Energy balance error: "
        f"{abs(Q_sep - Q_conv - Q_rad):.3e} W"
    )

    constraints = evaluate_thermal_constraints(
        T_wg_sep,
        T_wc_sep,
        T_c_sep,
        T_wall_allow=1000.0,
        T_coolant_allow=600.0
    )

    print("\n--- Thermal Design Constraints ---")

    for key, value in constraints.items():
        if isinstance(value, (bool, np.bool_)):
            print(f"{key}: {value}")
        else:
            print(f"{key}: {value:.2f}")

    mass_flow_values = [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]

    sensitivity = run_mass_flow_sensitivity(
        x,
        dA_wall,
        T_aw,
        data["T"],
        h_g,
        mass_flow_values,
        T_c_in=T_c_in,
        c_p=c_p,
        h_c=h_c,
        L_w=L_w,
        k_w=k_w,
        epsilon_eff=epsilon_eff
    )

    print("\n--- Coolant Mass-Flow Sensitivity ---")

    print(
        f"{'Flow (kg/s)':>12} "
        f"{'T_out (K)':>12} "
        f"{'Wall max (K)':>14} "
        f"{'Heat (kW)':>12} "
        f"{'Wall margin':>13} "
        f"{'Wall pass':>10}"
    )

    for result in sensitivity:
        print(
            f"{result['m_dot_c']:12.2f} "
            f"{result['T_c_out']:12.2f} "
            f"{result['T_wg_max']:14.2f} "
            f"{result['Q_total'] / 1e3:12.2f} "
            f"{result['wall_margin']:13.2f} "
            f"{str(result['wall_pass']):>10}"
        )
    
    hc_values = [
        5000.0,
        10000.0,
        15000.0,
        20000.0,
        30000.0,
        50000.0
    ]

    hc_sensitivity = run_hc_sensitivity(
        x,
        dA_wall,
        T_aw,
        data["T"],
        h_g,
        hc_values,
        T_c_in=T_c_in,
        m_dot_c=m_dot_c,
        c_p=c_p,
        L_w=L_w,
        k_w=k_w,
        epsilon_eff=epsilon_eff
    )

    print("\n--- Coolant Heat-Transfer Coefficient Sensitivity ---")

    print(
        f"{'h_c (W/m²K)':>14} "
        f"{'T_out (K)':>12} "
        f"{'Wall max (K)':>14} "
        f"{'Heat (kW)':>12} "
        f"{'Wall margin':>13} "
        f"{'Wall pass':>10}"
    )

    for result in hc_sensitivity:
        print(
            f"{result['h_c']:14.0f} "
            f"{result['T_c_out']:12.2f} "
            f"{result['T_wg_max']:14.2f} "
            f"{result['Q_total'] / 1e3:12.2f} "
            f"{result['wall_margin']:13.2f} "
            f"{str(result['wall_pass']):>10}"
        )

    kw_values = [
        20.0,
        50.0,
        100.0,
        200.0,
        350.0,
        500.0
    ]

    kw_sensitivity = run_kw_sensitivity(
        x,
        dA_wall,
        T_aw,
        data["T"],
        h_g,
        kw_values,
        T_c_in=T_c_in,
        m_dot_c=m_dot_c,
        c_p=c_p,
        h_c=h_c,
        L_w=L_w,
        epsilon_eff=epsilon_eff
    )

    print("\n--- Wall Conductivity Sensitivity ---")

    print(
        f"{'k_w (W/mK)':>12} "
        f"{'T_out (K)':>12} "
        f"{'Wall max (K)':>14} "
        f"{'Heat (kW)':>12} "
        f"{'Wall margin':>13} "
        f"{'Wall pass':>10}"
    )

    for result in kw_sensitivity:
        print(
            f"{result['k_w']:12.0f} "
            f"{result['T_c_out']:12.2f} "
            f"{result['T_wg_max']:14.2f} "
            f"{result['Q_total'] / 1e3:12.2f} "
            f"{result['wall_margin']:13.2f} "
            f"{str(result['wall_pass']):>10}"
        )

    convergence = run_grid_convergence(
        x,
        data["r"],
        T_aw,
        data["T"],
        h_g,
        factors=(1, 2, 4, 8),
        T_c_in=T_c_in,
        m_dot_c=m_dot_c,
        c_p=c_p,
        h_c=h_c,
        L_w=L_w,
        k_w=k_w,
        epsilon_eff=epsilon_eff
    )

    print("\n--- Grid Convergence Study ---")

    print(
        f"{'Points':>8} "
        f"{'T_out (K)':>12} "
        f"{'Wall max (K)':>14} "
        f"{'Heat (kW)':>12}"
    )

    for result in convergence:
        print(
            f"{result['points']:8d} "
            f"{result['T_c_out']:12.4f} "
            f"{result['T_wg_max']:14.4f} "
            f"{result['Q_total'] / 1e3:12.4f}"
        )

    print("\n--- Finest Grid Differences ---")

    coarse = convergence[-2]
    fine = convergence[-1]

    print(
        f"Coolant outlet difference: "
        f"{abs(fine['T_c_out'] - coarse['T_c_out']):.6f} K"
    )

    print(
        f"Maximum wall temperature difference: "
        f"{abs(fine['T_wg_max'] - coarse['T_wg_max']):.6f} K"
    )

    validation = generate_validation_report(convergence)
    
    # Radiation diagnostics
    negative_radiation_points = np.count_nonzero(q_rad < 0)

    peak_conv_index = np.argmax(q_conv)
    peak_rad_index = np.argmax(q_rad)
    peak_total_index = np.argmax(q_sep)

    print("\n--- Radiation Diagnostics ---")

    print(
        f"Peak convective heat flux: "
        f"{q_conv[peak_conv_index] / 1e6:.4f} MW/m²"
    )

    print(
        f"Peak convective location: "
        f"{x[peak_conv_index]:.4f} m"
    )

    print(
        f"Peak radiative heat flux: "
        f"{q_rad[peak_rad_index] / 1e6:.4f} MW/m²"
    )

    print(
        f"Peak radiative location: "
        f"{x[peak_rad_index]:.4f} m"
    )

    print(
        f"Peak total heat flux location: "
        f"{x[peak_total_index]:.4f} m"
    )

    print(
        f"Minimum radiative heat flux: "
        f"{np.min(q_rad) / 1e6:.4f} MW/m²"
    )

    print(
        f"Number of negative radiation points: "
        f"{negative_radiation_points}"
    )

    print(
        f"Maximum local flux balance error: "
        f"{np.max(np.abs(q_sep - q_conv - q_rad)):.3e} W/m²"
    )

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)


    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 8),
        sharex=True
    )

    # Nozzle geometry
    ax1.plot(x, data["r"] * 1000, linewidth=2)
    ax1.plot(x, -data["r"] * 1000, linewidth=2)

    ax1.fill_between(
        x,
        -data["r"] * 1000,
        data["r"] * 1000,
        alpha=0.15
    )

    ax1.axvline(0, color="gray", linestyle="--")
    ax1.set_ylabel("Radius (mm)")
    ax1.set_title("Nozzle Geometry")
    ax1.grid(True)

    # Heat flux
    ax2.plot(
        x, q_flux / 1e6,
        linewidth=2,
        label="Wall heat flux"
    )

    ax2.axvline(
        x[peak_flux_index],
        color="red",
        linestyle="--",
        label="Peak heat flux"
    )

    ax2.set_xlabel("Axial position (m)")
    ax2.set_ylabel("Heat flux (MW/m²)")
    ax2.set_title("Integrated Nozzle Heat Flux")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()

    plt.savefig(
        results_dir / "integrated_nozzle_heat_flux.png",
        dpi=300
    )

    plt.show()

    plt.figure(figsize=(10, 6))

    plt.plot(
        x, T_aw,
        label="Adiabatic wall temperature",
        linewidth=2
    )

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

    plt.plot(
        x, T_c,
        label="Coolant bulk temperature",
        linewidth=2
    )

    plt.axvline(
        0,
        color="gray",
        linestyle="--",
        label="Throat"
    )

    plt.xlabel("Axial position (m)")
    plt.ylabel("Temperature (K)")
    plt.title("Integrated Nozzle Thermal Distribution")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        results_dir / "integrated_nozzle_temperatures.png",
        dpi=300
    )

    plt.show()

    plt.figure(figsize=(10, 5))

    plt.plot(
        x, P_c / 1e6,
        linewidth=2,
        label="Counterflow coolant pressure"
    )

    plt.axvline(
        0,
        color="gray",
        linestyle="--",
        label="Throat"
    )

    plt.xlabel("Axial position (m)")
    plt.ylabel("Coolant pressure (MPa)")
    plt.title("Integrated Nozzle Coolant Pressure")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        results_dir / "integrated_nozzle_pressure.png",
        dpi=300
    )

    plt.show()

    # Extract sensitivity results
    flows = np.array([
        result["m_dot_c"] for result in sensitivity
    ])

    wall_temps = np.array([
        result["T_wg_max"] for result in sensitivity
    ])

    coolant_temps = np.array([
        result["T_c_out"] for result in sensitivity
    ])

    # Plot thermal sensitivity
    fig, axes = plt.subplots(2, 1, figsize=(9, 9))

    axes[0].plot(
        flows, wall_temps,
        marker="o",
        linewidth=2,
        label="Maximum hot-side wall temperature"
    )

    axes[0].axhline(
        1000.0,
        linestyle="--",
        color="red",
        label="Provisional wall limit"
    )

    axes[0].set_xlabel("Coolant mass flow (kg/s)")
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title("Wall Temperature vs Coolant Mass Flow")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        flows, coolant_temps,
        marker="o",
        linewidth=2,
        label="Coolant outlet temperature"
    )

    axes[1].axhline(
        600.0,
        linestyle="--",
        color="red",
        label="Provisional coolant limit"
    )

    axes[1].set_xlabel("Coolant mass flow (kg/s)")
    axes[1].set_ylabel("Temperature (K)")
    axes[1].set_title("Coolant Temperature vs Coolant Mass Flow")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()

    plt.savefig(
        results_dir / "mass_flow_sensitivity.png",
        dpi=300
    )

    plt.show()

    # Plot separated heat-transfer mechanisms
    plt.figure(figsize=(10, 6))

    plt.plot(
        x,
        q_conv / 1e6,
        linewidth=2,
        label="Convective heat flux"
    )

    plt.plot(
        x,
        q_rad / 1e6,
        linewidth=2,
        label="Radiative heat flux"
    )

    plt.plot(
        x,
        q_sep / 1e6,
        linewidth=2,
        linestyle="--",
        label="Total heat flux"
    )

    plt.axhline(
        0,
        color="black",
        linewidth=0.8
    )

    plt.axvline(
        0,
        color="gray",
        linestyle=":",
        label="Throat"
    )

    plt.xlabel("Axial position (m)")
    plt.ylabel("Heat flux (MW/m²)")
    plt.title("Nozzle Convection and Radiation Heat Flux")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        results_dir / "nozzle_convection_radiation.png",
        dpi=300
    )

    plt.show()
