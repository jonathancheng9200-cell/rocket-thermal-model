
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from thermal_resistance import thermal_resistance_radiation

from thermal_radiation_separated import thermal_resistance_separated

def coolant_temperature_solver(
    x, T_g, h_g,
    T_c_in, m_dot_c, c_p,
    P_h, h_c, L_w, k_w, epsilon_eff
):

    x = np.asarray(x, dtype=float)
    T_g = np.asarray(T_g, dtype=float)
    h_g = np.asarray(h_g, dtype=float)

    # Validate axial grid and gas profiles
    if x.ndim != 1 or T_g.shape != x.shape or h_g.shape != x.shape:
        raise ValueError("x, T_g, and h_g must be matching 1D arrays.")

    if len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("Axial grid must contain at least two finite points.")

    if not np.all(np.diff(x) > 0):
        raise ValueError("Axial positions must be strictly increasing.")

    # Validate coolant parameters
    if not np.all(np.isfinite([T_c_in, m_dot_c, c_p, P_h])):
        raise ValueError("Coolant inputs must be finite.")

    if T_c_in <= 0 or m_dot_c <= 0 or c_p <= 0 or P_h <= 0:
        raise ValueError("Coolant inputs must be positive.")

    N = len(x)

    # Initialize arrays
    T_c = np.zeros(N)
    q_flux = np.zeros(N)
    T_wg = np.zeros(N)
    T_wc = np.zeros(N)

    T_c[0] = T_c_in

    # March in the coolant flow direction
    for i in range(N - 1):

        q_flux[i], T_wg[i], T_wc[i] = (
            thermal_resistance_radiation(
                T_g[i],
                T_c[i],
                h_g[i],
                h_c,
                L_w,
                k_w,
                epsilon_eff
            )
        )

        ds = x[i + 1] - x[i]

        # Heat absorbed in this segment
        dQ = q_flux[i] * P_h * ds

        # Coolant temperature update
        T_c[i + 1] = (
            T_c[i] + dQ / (m_dot_c * c_p)
        )

    # Calculate thermal state at final location
    q_flux[-1], T_wg[-1], T_wc[-1] = (
        thermal_resistance_radiation(
            T_g[-1],
            T_c[-1],
            h_g[-1],
            h_c,
            L_w,
            k_w,
            epsilon_eff
        )
    )

    return T_c, q_flux, T_wg, T_wc

def coolant_pressure_solver(
    x, P_in, rho_c, V_c, D_h, f_D
):

    x = np.asarray(x, dtype=float)

    # Validate inputs
    if x.ndim != 1 or len(x) < 2:
        raise ValueError("x must be a 1D array with at least 2 points.")

    if not np.all(np.isfinite(x)):
        raise ValueError("Axial positions must be finite.")

    if not np.all(np.diff(x) > 0):
        raise ValueError("Axial positions must be strictly increasing.")

    parameters = [P_in, rho_c, V_c, D_h, f_D]

    if not np.all(np.isfinite(parameters)):
        raise ValueError("Pressure model inputs must be finite.")

    if P_in <= 0 or rho_c <= 0 or V_c < 0 or D_h <= 0 or f_D < 0:
        raise ValueError("Invalid coolant pressure model inputs.")

    # Initialize pressure array
    P_c = np.zeros(len(x))
    P_c[0] = P_in

    # March along coolant flow direction
    for i in range(len(x) - 1):

        ds = x[i + 1] - x[i]

        dP = (
            f_D
            * (ds / D_h)
            * (rho_c * V_c**2 / 2)
        )

        P_c[i + 1] = P_c[i] - dP

        if P_c[i + 1] <= 0:
            raise ValueError("Predicted coolant pressure is nonpositive.")

    return P_c

def regenerative_cooling_solver(
    x, T_g, h_g,
    T_c_in, m_dot_c, c_p,
    P_h, h_c, L_w, k_w, epsilon_eff,
    P_in, rho_c, V_c, D_h, f_D
):

    # Calculate coolant temperature and wall conditions
    T_c, q_flux, T_wg, T_wc = coolant_temperature_solver(
        x, T_g, h_g,
        T_c_in, m_dot_c, c_p,
        P_h, h_c, L_w, k_w, epsilon_eff
    )

    # Calculate coolant pressure distribution
    P_c = coolant_pressure_solver(
        x, P_in, rho_c, V_c, D_h, f_D
    )

    return T_c, P_c, q_flux, T_wg, T_wc

def rectangular_channel_geometry(
    n_channels, width, height, m_dot_c, rho_c
):

    # Validate inputs
    if isinstance(n_channels, bool) or not isinstance(
        n_channels, (int, np.integer)
    ) or n_channels <= 0:
        raise ValueError("Number of channels must be a positive integer.")

    parameters = [width, height, m_dot_c, rho_c]

    if not np.all(np.isfinite(parameters)):
        raise ValueError("Channel parameters must be finite.")

    if any(value <= 0 for value in parameters):
        raise ValueError("Channel parameters must be positive.")

    # Area of one channel
    A_channel = width * height

    # Total flow area
    A_total = n_channels * A_channel

    # Wetted perimeter of one channel
    P_wet = 2 * (width + height)

    # Hydraulic diameter
    D_h = 4 * A_channel / P_wet

    # Coolant velocity from continuity
    V_c = m_dot_c / (rho_c * A_total)

    return A_total, D_h, V_c

def coolant_temperature_solver_directional(
    x, T_g, h_g,
    T_c_in, m_dot_c, c_p,
    P_h, h_c, L_w, k_w, epsilon_eff,
    direction="coflow"
):

    x = np.asarray(x, dtype=float)
    T_g = np.asarray(T_g, dtype=float)
    h_g = np.asarray(h_g, dtype=float)

    if direction not in ("coflow", "counterflow"):
        raise ValueError(
            "Direction must be 'coflow' or 'counterflow'."
        )

    if x.ndim != 1 or T_g.shape != x.shape or h_g.shape != x.shape:
        raise ValueError("x, T_g, and h_g must be matching 1D arrays.")

    if len(x) < 2 or not np.all(np.isfinite(x)):
        raise ValueError("Axial grid must contain at least two finite points.")

    if not np.all(np.diff(x) > 0):
        raise ValueError("Axial grid must be strictly increasing.")

    # Reuse physical validation from the existing solver
    if direction == "coflow":
        return coolant_temperature_solver(
            x, T_g, h_g,
            T_c_in, m_dot_c, c_p,
            P_h, h_c, L_w, k_w, epsilon_eff
        )

    # Reverse spatial inputs for coolant marching
    x_flow = x[::-1]
    T_g_flow = T_g[::-1]
    h_g_flow = h_g[::-1]

    # Coolant distance always increases along the flow path
    s = x[-1] - x_flow

    # Solve in coolant flow direction
    T_c_flow, q_flow, T_wg_flow, T_wc_flow = (
        coolant_temperature_solver(
            s, T_g_flow, h_g_flow,
            T_c_in, m_dot_c, c_p,
            P_h, h_c, L_w, k_w, epsilon_eff
        )
    )

    # Return arrays in original chamber-to-exit order
    return (
        T_c_flow[::-1],
        q_flow[::-1],
        T_wg_flow[::-1],
        T_wc_flow[::-1]
    )

def coolant_pressure_solver_directional(
    x, P_in, rho_c, V_c, D_h, f_D,
    direction="coflow"
):

    x = np.asarray(x, dtype=float)

    if direction not in ("coflow", "counterflow"):
        raise ValueError(
            "Direction must be 'coflow' or 'counterflow'."
        )

    # Coflow: inlet at x[0]
    if direction == "coflow":
        return coolant_pressure_solver(
            x, P_in, rho_c, V_c, D_h, f_D
        )

    # Counterflow: inlet at x[-1]
    x_flow = x[::-1]

    # Distance increasing along coolant flow
    s = x[-1] - x_flow

    # March pressure in coolant flow direction
    P_flow = coolant_pressure_solver(
        s, P_in, rho_c, V_c, D_h, f_D
    )

    # Return pressure in original axial order
    return P_flow[::-1]

def coolant_temperature_solver_geometry(
    x, dA_wall, T_g, h_g,
    T_c_in, m_dot_c, c_p,
    h_c, L_w, k_w, epsilon_eff,
    direction="counterflow"
):

    x = np.asarray(x, dtype=float)
    dA_wall = np.asarray(dA_wall, dtype=float)
    T_g = np.asarray(T_g, dtype=float)
    h_g = np.asarray(h_g, dtype=float)

    if direction not in ("coflow", "counterflow"):
        raise ValueError("Invalid coolant flow direction.")

    n = len(x)

    if (
        x.ndim != 1
        or T_g.shape != (n,)
        or h_g.shape != (n,)
        or dA_wall.shape != (n - 1,)
    ):
        raise ValueError("Inconsistent axial array dimensions.")

    if n < 2 or not np.all(np.diff(x) > 0):
        raise ValueError("x must be strictly increasing.")

    if not all(
        np.all(np.isfinite(arr))
        for arr in (x, dA_wall, T_g, h_g)
    ):
        raise ValueError("Inputs must be finite.")

    if np.any(dA_wall <= 0):
        raise ValueError("Segment wall areas must be positive.")

    if np.any(h_g <= 0):
        raise ValueError("Gas heat-transfer coefficients must be positive.")

    if any(
        not np.isfinite(v) or v <= 0
        for v in (T_c_in, m_dot_c, c_p, h_c, L_w, k_w)
    ):
        raise ValueError("Thermal parameters must be positive and finite.")

    if not np.isfinite(epsilon_eff) or not 0 <= epsilon_eff <= 1:
        raise ValueError("Effective emissivity must be between 0 and 1.")

    # Reuse the existing local thermal solver
    # Adjust this import to match your function name if needed
    from thermal_resistance import thermal_resistance_radiation

    T_c = np.zeros(n)
    q_flux = np.zeros(n)
    T_wg = np.zeros(n)
    T_wc = np.zeros(n)

    if direction == "coflow":
        indices = range(n)
        inlet_index = 0
    else:
        indices = range(n - 1, -1, -1)
        inlet_index = n - 1

    T_c[inlet_index] = T_c_in

    indices = list(indices)

    for step, i in enumerate(indices):

        q, Twg, Twc = thermal_resistance_radiation(
            T_g[i],
            T_c[i],
            h_g[i],
            h_c,
            L_w,
            k_w,
            epsilon_eff
        )

        q_flux[i] = q
        T_wg[i] = Twg
        T_wc[i] = Twc

        # Last point has no following segment
        if step == n - 1:
            continue

        next_i = indices[step + 1]

        # Segment between adjacent grid points
        segment_index = min(i, next_i)

        # Heat transferred through this wall segment
        dQ = q * dA_wall[segment_index]

        # Coolant temperature increase
        dT = dQ / (m_dot_c * c_p)

        T_c[next_i] = T_c[i] + dT

    return T_c, q_flux, T_wg, T_wc

def coolant_temperature_solver_separated(
    x, dA_wall, T_aw, T_rad, h_g,
    T_c_in, m_dot_c, c_p,
    h_c, L_w, k_w, epsilon_eff,
    direction="counterflow"
):

    x = np.asarray(x, dtype=float)
    dA_wall = np.asarray(dA_wall, dtype=float)
    T_aw = np.asarray(T_aw, dtype=float)
    T_rad = np.asarray(T_rad, dtype=float)
    h_g = np.asarray(h_g, dtype=float)

    if direction not in ("coflow", "counterflow"):
        raise ValueError("Invalid coolant flow direction.")

    n = len(x)

    if (
        x.ndim != 1
        or n < 2
        or T_aw.shape != (n,)
        or T_rad.shape != (n,)
        or h_g.shape != (n,)
        or dA_wall.shape != (n - 1,)
    ):
        raise ValueError("Inconsistent axial array dimensions.")

    if not np.all(np.isfinite(x)) or not np.all(np.diff(x) > 0):
        raise ValueError("x must be finite and strictly increasing.")

    if not all(
        np.all(np.isfinite(arr))
        for arr in (dA_wall, T_aw, T_rad, h_g)
    ):
        raise ValueError("Input arrays must be finite.")

    if np.any(dA_wall <= 0):
        raise ValueError("Segment wall areas must be positive.")

    if np.any(T_aw <= 0) or np.any(T_rad <= 0) or np.any(h_g <= 0):
        raise ValueError("Gas temperatures and coefficients must be positive.")

    if any(
        not np.isfinite(v) or v <= 0
        for v in (T_c_in, m_dot_c, c_p, h_c, L_w, k_w)
    ):
        raise ValueError("Thermal parameters must be positive and finite.")

    if not np.isfinite(epsilon_eff) or not 0 <= epsilon_eff <= 1:
        raise ValueError("Emissivity must be between 0 and 1.")

    T_c = np.zeros(n)
    q_total = np.zeros(n)
    q_conv = np.zeros(n)
    q_rad = np.zeros(n)
    T_wg = np.zeros(n)
    T_wc = np.zeros(n)

    if direction == "coflow":
        indices = list(range(n))
    else:
        indices = list(range(n - 1, -1, -1))

    inlet_index = indices[0]
    T_c[inlet_index] = T_c_in

    for step, i in enumerate(indices):

        q, Twg, Twc, qc, qr = thermal_resistance_separated(
            T_aw[i],
            T_rad[i],
            T_c[i],
            h_g[i],
            h_c,
            L_w,
            k_w,
            epsilon_eff
        )

        q_total[i] = q
        q_conv[i] = qc
        q_rad[i] = qr
        T_wg[i] = Twg
        T_wc[i] = Twc

        if step == n - 1:
            continue

        next_i = indices[step + 1]
        segment_index = min(i, next_i)

        dQ = q * dA_wall[segment_index]

        T_c[next_i] = T_c[i] + dQ / (m_dot_c * c_p)

    return T_c, q_total, T_wg, T_wc, q_conv, q_rad

if __name__ == "__main__":

    # Axial grid
    x = np.linspace(0, 0.5, 100)

    # Prescribed gas profiles
    T_g = 3000 - 1000 * (x / 0.5)

    h_g = 4000 + 6000 * np.exp(
        -((x - 0.25) / 0.08)**2
    )

    # Coolant properties
    T_c_in = 300.0
    m_dot_c = 2.0
    c_p = 3000.0
    P_h = 0.10

    # Wall properties
    h_c = 15000.0
    L_w = 0.003
    k_w = 350.0
    epsilon_eff = 0.20

    # Run coupled coolant solver
    T_c, q_flux, T_wg, T_wc = coolant_temperature_solver(
        x, T_g, h_g,
        T_c_in, m_dot_c, c_p,
        P_h, h_c, L_w, k_w, epsilon_eff
    )

    # Calculate total absorbed heat
    Q_total = m_dot_c * c_p * (T_c[-1] - T_c[0])

    print(f"Coolant inlet temperature: {T_c[0]:.2f} K")
    print(f"Coolant outlet temperature: {T_c[-1]:.2f} K")
    print(f"Total absorbed heat: {Q_total / 1e3:.2f} kW")
    print(f"Maximum wall heat flux: {np.max(q_flux) / 1e6:.4f} MW/m²")
    print(f"Maximum hot-side wall temperature: {np.max(T_wg):.2f} K")

    # Coolant pressure parameters
    P_in = 5.0e6
    rho_c = 1000.0
    V_c = 4.0
    D_h = 0.005
    f_D = 0.020

    # Run pressure solver
    P_c = coolant_pressure_solver(
        x, P_in, rho_c, V_c, D_h, f_D
    )

    # Calculate total pressure loss
    delta_P = P_c[0] - P_c[-1]

    print("\n--- Coolant Pressure Results ---")
    print(f"Inlet pressure: {P_c[0] / 1e6:.4f} MPa")
    print(f"Outlet pressure: {P_c[-1] / 1e6:.4f} MPa")
    print(f"Total pressure loss: {delta_P / 1e3:.2f} kPa")

    # Integrated regenerative cooling simulation
    T_c_int, P_c_int, q_int, T_wg_int, T_wc_int = (
        regenerative_cooling_solver(
            x, T_g, h_g,
            T_c_in, m_dot_c, c_p,
            P_h, h_c, L_w, k_w, epsilon_eff,
            P_in, rho_c, V_c, D_h, f_D
        )
    )

    print("\n--- Integrated Cooling Results ---")
    print(f"Coolant outlet temperature: {T_c_int[-1]:.2f} K")
    print(f"Coolant outlet pressure: {P_c_int[-1] / 1e6:.4f} MPa")
    print(f"Maximum wall heat flux: {np.max(q_int) / 1e6:.4f} MW/m²")
    print(f"Maximum hot-side wall temperature: {np.max(T_wg_int):.2f} K")

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))

    plt.plot(
        x, T_c_int,
        linewidth=2,
        label="Coolant bulk temperature"
    )

    plt.xlabel("Axial position (m)")
    plt.ylabel("Temperature (K)")
    plt.title("Regenerative Cooling: Coolant Temperature")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        results_dir / "coolant_temperature.png",
        dpi=300
    )

    plt.show()

    plt.figure(figsize=(9, 5))

    plt.plot(
        x, P_c_int / 1e6,
        linewidth=2,
        label="Coolant pressure"
    )

    plt.xlabel("Axial position (m)")
    plt.ylabel("Pressure (MPa)")
    plt.title("Regenerative Cooling: Coolant Pressure")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        results_dir / "coolant_pressure.png",
        dpi=300
    )

    plt.show()

    # Rectangular cooling channel geometry
    n_channels = 50
    width = 0.005
    height = 0.002

    A_total, D_h_geom, V_c_geom = rectangular_channel_geometry(
        n_channels, width, height, m_dot_c, rho_c
    )

    print("\n--- Cooling Channel Geometry ---")
    print(f"Total flow area: {A_total:.6f} m²")
    print(f"Hydraulic diameter: {D_h_geom * 1000:.4f} mm")
    print(f"Coolant velocity: {V_c_geom:.4f} m/s")

    # Recalculate pressure using geometry-based values
    P_c_geom = coolant_pressure_solver(
        x, P_in, rho_c, V_c_geom, D_h_geom, f_D
    )

    delta_P_geom = P_c_geom[0] - P_c_geom[-1]

    print(f"Updated pressure loss: {delta_P_geom / 1e3:.2f} kPa")
    print(f"Updated outlet pressure: {P_c_geom[-1] / 1e6:.4f} MPa")

    # Counterflow coolant simulation
    T_c_counter, q_counter, T_wg_counter, T_wc_counter = (
        coolant_temperature_solver_directional(
            x, T_g, h_g,
            T_c_in, m_dot_c, c_p,
            P_h, h_c, L_w, k_w, epsilon_eff,
            direction="counterflow"
        )
    )

    print("\n--- Counterflow Cooling Results ---")
    print(f"Coolant inlet temperature: {T_c_counter[-1]:.2f} K")
    print(f"Coolant outlet temperature: {T_c_counter[0]:.2f} K")
    print(f"Maximum wall heat flux: {np.max(q_counter) / 1e6:.4f} MW/m²")
    print(f"Maximum hot-side wall temperature: {np.max(T_wg_counter):.2f} K")

    print("\n--- Coflow vs Counterflow ---")

    print(
        f"Coflow outlet temperature: "
        f"{T_c_int[-1]:.2f} K"
    )

    print(
        f"Counterflow outlet temperature: "
        f"{T_c_counter[0]:.2f} K"
    )

    print(
        f"Coflow peak wall temperature: "
        f"{np.max(T_wg_int):.2f} K"
    )

    print(
        f"Counterflow peak wall temperature: "
        f"{np.max(T_wg_counter):.2f} K"
    )

    # Counterflow pressure using channel geometry
    P_c_counter = coolant_pressure_solver_directional(
        x,
        P_in,
        rho_c,
        V_c_geom,
        D_h_geom,
        f_D,
        direction="counterflow"
    )

    print("\n--- Counterflow Pressure Results ---")

    print(
        f"Counterflow inlet pressure: "
        f"{P_c_counter[-1] / 1e6:.4f} MPa"
    )

    print(
        f"Counterflow outlet pressure: "
        f"{P_c_counter[0] / 1e6:.4f} MPa"
    )

    print(
        f"Counterflow pressure loss: "
        f"{(P_c_counter[-1] - P_c_counter[0]) / 1e3:.2f} kPa"
    )

    # Coflow pressure using the same channel geometry
    P_c_coflow = coolant_pressure_solver_directional(
        x,
        P_in,
        rho_c,
        V_c_geom,
        D_h_geom,
        f_D,
        direction="coflow"
    )

    plt.figure(figsize=(9, 5))

    plt.plot(
        x, P_c_coflow / 1e6,
        linewidth=2,
        label="Coflow"
    )

    plt.plot(
        x, P_c_counter / 1e6,
        linewidth=2,
        label="Counterflow"
    )

    plt.xlabel("Axial position (m)")
    plt.ylabel("Coolant pressure (MPa)")
    plt.title("Coflow vs Counterflow: Pressure Distribution")

    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        results_dir / "coflow_vs_counterflow_pressure.png",
        dpi=300
    )

    plt.show()

