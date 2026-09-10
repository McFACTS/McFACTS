"""Validate analytic sampling against numerical integrals of the density."""
import numpy as np
import pytest
from scipy.integrate import quad

from mcfacts.setup.setupdiskblackholes import (
    setup_disk_blackholes_location_NSC_powerlaw_optimized as sample_locations,
)


class FixedDraws:
    """Supply known probabilities so the CDF can be checked without sampling noise."""

    def __init__(self, values):
        self.values = np.asarray(values)

    def random(self, size):
        assert size == self.values.size
        return self.values.copy()


@pytest.mark.parametrize("volume_scaling", [False, True])
@pytest.mark.parametrize("critical_radius", [2., 6., 5000., 50000., 100000.])
@pytest.mark.parametrize("indices", [(1.75, 2.5), (1., 3.), (3., 1.),
                                     (1. + 1.e-10, 3. - 1.e-10)])
def test_quantiles_match_integrated_density(volume_scaling, critical_radius, indices):
    probabilities = np.array([0., 0.001, 0.1, 0.3, 0.5, 0.8, 0.999])
    locations = sample_locations(
        len(probabilities), 50000., 6., 1.e8, critical_radius / 2.e5,
        *indices, FixedDraws(probabilities), volume_scaling=volume_scaling,
    )

    def integral(end):
        def density(radius):
            index = indices[0] if radius <= critical_radius else indices[1]
            return (radius / critical_radius)**(-index) * (
                radius**2 if volume_scaling else 1.
            )

        # Explicitly split the independent quadrature at the density break.
        points = [critical_radius] if 6. < critical_radius < end else None
        return quad(density, 6., end, points=points, epsabs=1.e-10, epsrel=1.e-10)[0]

    normalization = integral(50000.)
    measured = [integral(location) / normalization for location in locations]
    np.testing.assert_allclose(measured, probabilities, rtol=1.e-8, atol=1.e-10)
    assert np.all((locations >= 6.) & (locations <= 50000.))


@pytest.mark.parametrize("count", [0, 1, 10000])
def test_reproducible_bounded_draws(count):
    args = (count, 50000., 6., 1.e8, 0.025, 1.75, 2.5)
    first = sample_locations(*args, np.random.default_rng(123))
    second = sample_locations(*args, np.random.default_rng(123))
    assert first.shape == (count,)
    assert first.dtype == np.float64
    np.testing.assert_array_equal(first, second)
    assert np.all((first >= 6.) & (first <= 50000.))


@pytest.mark.parametrize("overrides", [
    {"disk_inner_stable_circ_orb": 0.},
    {"disk_inner_stable_circ_orb": 50000.},
    {"disk_radius_outer": 5.},
    {"smbh_mass": 0.},
    {"nsc_radius_crit": -1.},
    {"nsc_density_index_inner": np.nan},
    {"nsc_density_index_outer": np.inf},
])
def test_invalid_distribution_parameters(overrides):
    parameters = dict(disk_bh_num=1, disk_radius_outer=50000.,
                      disk_inner_stable_circ_orb=6., smbh_mass=1.e8,
                      nsc_radius_crit=0.025, nsc_density_index_inner=1.75,
                      nsc_density_index_outer=2.5, random=np.random.default_rng(123))
    parameters.update(overrides)
    with pytest.raises(ValueError):
        sample_locations(**parameters)
