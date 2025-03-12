import setuptools

version_file = "__version__.py"

def get_version():

    with open(version_file, 'r') as file:
        _line = file.read()

    __version__ = _line.split("=")[-1].lstrip(" '").rstrip(" '\n")

    return __version__

PACKAGE_DATA = {
    "mcfacts": [
        "vis/mcfacts_figures.mplstyle",
        "inputs/data/sirko_goodman_aspect_ratio.txt",
        "inputs/data/thompson_etal_aspect_ratio.txt",
        "inputs/data/sirko_goodman_opacity.txt",
        "inputs/data/thompson_etal_opacity.txt",
        "inputs/data/sirko_goodman_surface_density.txt",
        "inputs/data/thompson_etal_surface_density.txt",
        "inputs/data/sirko_goodman_density.txt",
        "inputs/data/thompson_etal_density.txt",
        "inputs/data/sirko_goodman_omega.txt",
        "inputs/data/thompson_etal_omega.txt",
        "inputs/data/sirko_goodman_pressure_gradient.txt",
        "inputs/data/thompson_etal_pressure_gradient.txt",
        "inputs/data/sirko_goodman_sound_speed.txt",
        "inputs/data/thompson_etal_sound_speed.txt",
        "inputs/data/sirko_goodman_temperature.txt",
        "inputs/data/thompson_etal_temperature.txt",
        "inputs/data/stellar_grid/stellar_grid.txt",
        "inputs/data/model_choice.ini",
        "vis/data/O3-H1-C01_CLEAN_SUB60HZ-1262197260.0_sensitivity_strain_asd.txt",
        "vis/data/O3-L1-C01_CLEAN_SUB60HZ-1240573680.0_sensitivity_strain_asd.txt",
        "vis/data/O3-L1-C01_CLEAN_SUB60HZ-1262141640.0_sensitivity_strain_asd.txt",
    ],
}
# Thank you:
setuptools.setup(
    version = get_version(),
    package_data=PACKAGE_DATA,
)
