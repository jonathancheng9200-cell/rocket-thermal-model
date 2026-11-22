
import numpy as np


def estimate_first_order_limit(coarse, fine):
    """
    Richardson extrapolation for refinement ratio 2
    and assumed first-order convergence.
    """
    return 2.0 * fine - coarse


def generate_validation_report(convergence):

    if len(convergence) < 3:
        raise ValueError(
            "At least three grid levels are required."
        )

    coarse = convergence[-2]
    fine = convergence[-1]

    Tc_inf = estimate_first_order_limit(
        coarse["T_c_out"],
        fine["T_c_out"]
    )

    Tw_inf = estimate_first_order_limit(
        coarse["T_wg_max"],
        fine["T_wg_max"]
    )

    Tc_error = abs(Tc_inf - fine["T_c_out"])
    Tw_error = abs(Tw_inf - fine["T_wg_max"])

    print("\n--- Numerical Validation Report ---")

    print(f"Finest grid points: {fine['points']}")

    print(
        f"Extrapolated coolant outlet: "
        f"{Tc_inf:.4f} K"
    )

    print(
        f"Extrapolated maximum wall temperature: "
        f"{Tw_inf:.4f} K"
    )

    print(
        f"Estimated coolant temperature error: "
        f"{Tc_error:.6f} K"
    )

    print(
        f"Estimated wall temperature error: "
        f"{Tw_error:.6f} K"
    )

    print(
        f"Relative coolant error: "
        f"{100 * Tc_error / abs(Tc_inf):.5f}%"
    )

    print(
        f"Relative wall error: "
        f"{100 * Tw_error / abs(Tw_inf):.5f}%"
    )

    return {
        "Tc_inf": Tc_inf,
        "Tw_inf": Tw_inf,
        "Tc_error": Tc_error,
        "Tw_error": Tw_error
    }
