
import numpy as np

from nozzle_wall_geometry import nozzle_wall_segment_areas


def test_cylindrical_wall_area():

    x = np.linspace(0, 1, 101)
    radius = np.full_like(x, 0.05)

    dA, ds = nozzle_wall_segment_areas(x, radius)

    expected_area = 2 * np.pi * 0.05 * 1.0

    assert np.isclose(np.sum(dA), expected_area)
    assert np.isclose(np.sum(ds), 1.0)


def test_conical_frustum_area():

    x = np.array([0.0, 0.3])
    radius = np.array([0.05, 0.10])

    dA, ds = nozzle_wall_segment_areas(x, radius)

    expected_slant = np.sqrt(0.3**2 + 0.05**2)
    expected_area = np.pi * (0.05 + 0.10) * expected_slant

    assert np.isclose(ds[0], expected_slant)
    assert np.isclose(dA[0], expected_area)
