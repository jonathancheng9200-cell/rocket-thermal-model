
import numpy as np

from nozzle_interface import get_nozzle_data


def nozzle_wall_segment_areas(x, radius):

    x = np.asarray(x, dtype=float)
    radius = np.asarray(radius, dtype=float)

    if x.ndim != 1 or radius.ndim != 1:
        raise ValueError("Inputs must be 1D arrays.")

    if x.shape != radius.shape:
        raise ValueError("x and radius must have matching shapes.")

    if len(x) < 2:
        raise ValueError("At least two points are required.")

    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(radius)):
        raise ValueError("Inputs must be finite.")

    if not np.all(np.diff(x) > 0):
        raise ValueError("Axial coordinates must be strictly increasing.")

    if np.any(radius <= 0):
        raise ValueError("Nozzle radius must be positive.")

    dx = np.diff(x)
    dr = np.diff(radius)

    # Slant length of each nozzle segment
    ds = np.sqrt(dx**2 + dr**2)

    # Lateral surface area of each conical frustum
    dA = np.pi * (radius[:-1] + radius[1:]) * ds

    return dA, ds


if __name__ == "__main__":

    data = get_nozzle_data()

    dA, ds = nozzle_wall_segment_areas(
        data["x"],
        data["r"]
    )

    print("--- Nozzle Wall Geometry ---")
    print(f"Number of segments: {len(dA)}")
    print(f"Total wall area: {np.sum(dA):.6f} m²")
    print(f"Total slant length: {np.sum(ds):.6f} m")
    print(f"Minimum segment area: {np.min(dA):.6e} m²")
    print(f"Maximum segment area: {np.max(dA):.6e} m²")
