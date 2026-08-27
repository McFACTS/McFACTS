import warnings
from functools import cached_property
from types import NoneType
from typing import Any, TypeVar, Type

from mcfacts.inputs import ReadInputs
from mcfacts.utilities import unit_conversion

#### Setup
IGNORE_ARGS = [
    "print_version",
    "subcommand",
    "enable_profiling",
    "profiling_file",
]

class SettingsProperty:
    T = TypeVar("T")

    def __init__(self, name: str, category: str, value: Any, prop_type: Type[T] = str):
        assert type(value) == prop_type, f"SettingsProperty: the type of value ({type(value)}) must match prop_type {prop_type}"

        self._name = name
        self._category = category
        self._value = value
        self._prop_type = prop_type


    @property
    def name(self) -> str:
        return self._name


    @name.setter
    def name(self, value: str):
        self._name = value


    @property
    def category(self) -> str:
        return self._category


    @category.setter
    def category(self, value: str):
        self._category = value


    @property
    def value(self) -> T:
        return self._value


    @value.setter
    def value(self, value: T):
        self._value = value


    @property
    def prop_type(self) -> Type[T]:
        return self._prop_type


class StaticSettingsProperty(SettingsProperty):
    T = TypeVar("T")

    def __init__(self, name: str, category: str, value: Any, prop_type: Type[T] = str):
        super().__init__(name, category, value, prop_type)

        # Pre-cache properties
        _ = self.name
        _ = self.category
        _ = self.value
        _ = self.prop_type


    @cached_property
    def name(self) -> str:
        return self._name


    @cached_property
    def category(self) -> str:
        return self._category


    @cached_property
    def value(self) -> T:
        return self._value


    @cached_property
    def prop_type(self) -> Type[T]:
        return self._prop_type


class OptionalSettingsProperty(SettingsProperty):
    T = TypeVar("T")

    def __init__(self, name: str, category: str, value: Any, prop_type: Type[T] = str):
        super().__init__(name, category, value, prop_type)


DEFAULT_SETTINGS: list[SettingsProperty | StaticSettingsProperty] = [
        # IO Parameters
        SettingsProperty("verbose", "io", False, bool),
        SettingsProperty("show_timeline_progress", "io", False, bool),
        SettingsProperty("overwrite_files", "io", False, bool),
        SettingsProperty("save_state", "io", False, bool),
        SettingsProperty("save_each_timestep", "io", False, bool),
        SettingsProperty("output_dir", "io", "./runs", str),
        OptionalSettingsProperty("settings_file", "io", "", str),

        # Simulation Parameters
        SettingsProperty("active_timestep_duration_yr", "sim", 1.e4, float),
        SettingsProperty("active_timestep_num", "sim", 70, int),
        SettingsProperty("capture_time_yr", "sim", 1.e5, float),
        SettingsProperty("galaxy_num", "sim", 100, int),
        SettingsProperty("seed", "sim", 223849053863469657747974663531730220530, int),
        SettingsProperty("smbh_mass", "sim", 1.e8, float),
        StaticSettingsProperty("agn_redshift", "sim", 0.1, float),

        # Disk Parameters
        SettingsProperty("flag_use_pagn", "disk", True, bool),
        SettingsProperty("disk_model_name", "disk", "sirko_goodman", str),
        SettingsProperty("disk_radius_trap", "disk", 700.0, float),
        SettingsProperty("disk_radius_outer", "disk", 50000.0, float),
        SettingsProperty("disk_alpha_viscosity", "disk", 0.01, float),
        SettingsProperty("disk_aspect_ratio_avg", "disk", 0.03, float),
        SettingsProperty("disk_bh_torque_condition", "disk", 0.1, float),
        SettingsProperty("disk_bh_eddington_ratio", "disk", 1.0, float),
        SettingsProperty("disk_bh_orb_ecc_max_init", "disk", 0.9, float),
        SettingsProperty("disk_radius_capture_outer", "disk", 1.e3, float),
        SettingsProperty("disk_bh_pro_orb_ecc_crit", "disk", 0.01, float),
        SettingsProperty("inner_disk_outer_radius", "disk", 50.0, float),
        SettingsProperty("disk_radius_max_pc", "disk", 0., float),
        SettingsProperty("disk_inner_stable_circ_orb", "disk", 6.0, float),
        SettingsProperty("torque_prescription", "disk", "paardekooper", str),
        SettingsProperty("flag_phenom_turb", "disk", False, bool),
        SettingsProperty("phenom_turb_centroid", "disk", 0., float),
        SettingsProperty("phenom_turb_std_dev", "disk", 0.1, float),

        # Black Hole Parameters
        SettingsProperty("mass_pile_up", "bh", 35.0, float),
        SettingsProperty("initial_binary_orbital_ecc", "bh", 0.01, float),
        SettingsProperty("fraction_bin_retro", "bh", 0.0, float),
        SettingsProperty("flag_use_surrogate", "bh", False, bool),
        SettingsProperty("flag_use_spin_check", "bh", False, bool),
        SettingsProperty("mean_harden_energy_delta", "bh", 0.9, float),
        SettingsProperty("var_harden_energy_delta", "bh", 0.025, float),
        SettingsProperty("delta_energy_strong_mu", "bh", 0.1, float),
        SettingsProperty("delta_energy_strong_sigma", "bh", 0.02, float),
        SettingsProperty("flag_thermal_feedback", "bh", True, bool),
        SettingsProperty("flag_orb_ecc_damping", "bh", True, bool),
        SettingsProperty("flag_dynamic_enc", "bh", True, bool),
        SettingsProperty("flag_dynamics_sweep", "bh", True, bool),
        SettingsProperty("flag_enable_bondi", "bh", False, bool),
        SettingsProperty("bondi_fraction", "bh", 1e-5, float),
        SettingsProperty("stalling_separation", "bh", 0.0, float),
        SettingsProperty("gas_hardening_prescription", "bh", "baruteau", str),
        StaticSettingsProperty("disk_bh_eddington_mass_growth_rate", "bh", 2.3e-8, float),
        StaticSettingsProperty("disk_bh_spin_resolution_min", "bh", 0.02, float),
        StaticSettingsProperty("min_bbh_gw_separation", "bh", 2.0, float),

        # Star Parameters
        SettingsProperty("flag_add_stars", "star", False, bool),
        SettingsProperty("disk_star_mass_min_init", "star", 5.0, float),
        SettingsProperty("disk_star_mass_max_init", "star", 40.0, float),
        SettingsProperty("rstar_rhill_exponent_ratio", "star", 2.0, float),
        SettingsProperty("disk_star_scale_factor", "star", 1.e-3, float),
        SettingsProperty("flag_coalesce_initial_stars", "star", False, bool),
        SettingsProperty("flag_initial_stars_BH_immortal", "star", False, bool),
        SettingsProperty("disk_star_initial_mass_cutoff", "star", 298.0, float),

        # Nuclear Star Cluster Parameters
        SettingsProperty("nsc_radius_outer", "nsc", 5.0, float),
        SettingsProperty("nsc_mass", "nsc", 3.e7, float),
        SettingsProperty("nsc_radius_crit", "nsc", 0.25, float),
        SettingsProperty("nsc_ratio_bh_num_star_num", "nsc", 1.e-3, float),
        SettingsProperty("nsc_ratio_bh_mass_star_mass", "nsc", 10.0, float),
        SettingsProperty("nsc_density_index_inner", "nsc", 1.75, float),
        SettingsProperty("nsc_density_index_outer", "nsc", 2.5, float),
        SettingsProperty("nsc_imf_star_powerlaw_index", "nsc", 2.35, float),
        SettingsProperty("nsc_imf_bh_mode", "nsc", 10.0, float),
        SettingsProperty("nsc_imf_bh_powerlaw_index", "nsc", 2.0, float),
        SettingsProperty("nsc_imf_bh_mass_max", "nsc", 40.0, float),
        SettingsProperty("nsc_bh_spin_dist_mu", "nsc", 0.0, float),
        SettingsProperty("nsc_bh_spin_dist_sigma", "nsc", 0.1, float),
        SettingsProperty("nsc_spheroid_normalization", "nsc", 1.0, float),
        SettingsProperty("nsc_star_spin_dist_mu", "nsc", 100.0, float),
        SettingsProperty("nsc_star_spin_dist_sigma", "nsc", 20.0, float),
        SettingsProperty("nsc_star_metallicity_x_init", "nsc", 0.7064, float),
        SettingsProperty("nsc_star_metallicity_y_init", "nsc", 0.2735, float),
        SettingsProperty("nsc_star_metallicity_z_init", "nsc", 0.02, float),
        SettingsProperty("nsc_imf_bh_method", "nsc", "default", str),

        # Filing cabinet array names
        StaticSettingsProperty("bh_array_name", "arrays", "blackholes_unsort", str),
        StaticSettingsProperty("bh_inner_disk_array_name", "arrays", "blackholes_inner_disk", str),
        StaticSettingsProperty("bh_inner_gw_array_name", "arrays", "blackholes_inner_gw_only", str),
        StaticSettingsProperty("bh_prograde_array_name", "arrays", "blackholes_prograde", str),
        StaticSettingsProperty("bh_retrograde_array_name", "arrays", "blackholes_retrograde", str),
        StaticSettingsProperty("stars_prograde_array_name", "arrays", "stars_prograde", str),
        StaticSettingsProperty("stars_retrograde_array_name", "arrays", "stars_retrograde", str),
        StaticSettingsProperty("stars_merged_array_name", "arrays", "stars_merged", str),
        StaticSettingsProperty("bbh_array_name", "arrays", "blackholes_binary", str),
        StaticSettingsProperty("bbh_gw_array_name", "arrays", "blackholes_binary_gw", str),
        StaticSettingsProperty("bbh_merged_array_name", "arrays", "blackholes_merged", str),
        StaticSettingsProperty("emri_array_name", "arrays", "blackholes_emri", str),
        StaticSettingsProperty("star_array_name", "arrays", "stars_unsort", str),
        StaticSettingsProperty("bh_ejected_array_name", "arrays", "blackholes_ejected", str),
        StaticSettingsProperty("bbh_inter_array_name", "arrays", "blackholes_binary_inter", str),
    ]


class CategoryProxy:
    """
    A proxy object that holds settings for a single property category
    """

    def __init__(self, props: dict[str, Any] = None):
        self._props: dict[str, Any] = props if props is not None else {}

    def __getattr__(self, item: str) -> Any:
        if item.startswith("_"):
            raise AttributeError(f"No private attribute '{item}'")
        try:
            return self._props[item]
        except KeyError:
            raise AttributeError(f"No setting '{item}' in this category")

    def __repr__(self):
        return f"CategoryProxy({self._props})"


class SettingsManager:
    """
    Central management for simulation settings.
    """

    @staticmethod
    def _cast_str_to_int(prop: SettingsProperty, override: str) -> int:
        try:
            return int(override)
        except ValueError:
            pass

        try:
            float_val = float(override)
            warnings.warn(
                f"Setting '{prop.name}' expects an int, but got string '{override}' "
                f"which looks like a float. Precision may be lost after conversion to {int(float_val)}.",
                UserWarning,
                stacklevel=4,
            )
            return int(float_val)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Setting '{prop.name}' expects type int, "
                f"but could not cast string '{override}' to that type"
            ) from e


    @staticmethod
    def _cast_str_to_float(prop: SettingsProperty, override: str) -> float:
        try:
            return float(override)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Setting '{prop.name}' expects type float, "
                f"but could not cast string '{override}' to that type"
            ) from e


    @staticmethod
    def _cast_override(prop: SettingsProperty, override: Any) -> Any:
        """
        Attempts to safely cast an override value to the expected type of a SettingsProperty.

        Parameters
        ----------
        prop : SettingsProperty
            The property whose type the override should be cast to.
        override : Any
            The override value to cast.

        Returns
        -------
        Any
            The cast value.

        Raises
        ------
        TypeError
            If the override value cannot be cast to the expected type.
        """
        expected = prop.prop_type

        if type(override) == expected:
            return override

        if isinstance(override, str):
            if expected == bool:
                if override.lower() in ('yes', 'true', 't', 'y', '1'):
                    return True
                elif override.lower() in ('no', 'false', 'f', 'n', '0'):
                    return False
                else:
                    raise TypeError(
                        f"Setting '{prop.name}' expects a bool, but could not convert string '{override}' as a boolean."
                    )
            if expected == int:
                return SettingsManager._cast_str_to_int(prop, override)
            if expected == float:
                return SettingsManager._cast_str_to_float(prop, override)
            try:
                return expected(override)
            except (ValueError, TypeError) as e:
                raise TypeError(
                    f"Setting '{prop.name}' expects type {expected.__name__}, but could not cast string '{override}' to that type."
                ) from e

        if isinstance(override, float) and expected == int:
            warnings.warn(
                f"Setting '{prop.name}' expects an int, but got float {override}. Precision may be lost after conversion to {int(override)}.",
                UserWarning,
                stacklevel=3,
            )
            return int(override)

        if isinstance(override, int) and expected == float:
            return float(override)

        if isinstance(prop, OptionalSettingsProperty) and isinstance(override, NoneType):
            return override

        raise TypeError(
            f"Setting '{prop.name}' expects type {expected.__name__}, "
            f"but got {type(override).__name__}"
        )


    def __init__(self, settings_overrides: dict[str, Any] = None):
        """
        Initializes the SettingsManager with a dictionary of overrides.

        Parameters
        ----------
        settings_overrides : dict[str, Any]
            A dictionary containing user-provided overrides for specific settings.

        Raises
        ------
        TypeError
            If an override value is not the expected type and cannot be cast to it.
        """
        settings_overrides = dict() if settings_overrides is None else settings_overrides

        self.settings_finals: dict[str, Any] = {}
        category_props: dict[str, dict[str, Any]] = {}

        for prop in DEFAULT_SETTINGS:
            is_static = isinstance(prop, StaticSettingsProperty)
            override = settings_overrides.get(prop.name, prop.value)
            final_value = prop.value if is_static else self._cast_override(prop, override)

            self.settings_finals[prop.name] = final_value
            category_props.setdefault(prop.category, {})[prop.name] = final_value

        for key in settings_overrides.keys():
            if key in IGNORE_ARGS:
                continue
            if key in self.settings_finals:
                continue

            warnings.warn(
                f"Inputted settings dictionary contains {key}, which does not exist in the default settings."
                f"Custom settings must be added through the add_custom_settings method.",
                UserWarning,
                stacklevel=3,
            )

        self._categories = {cat: CategoryProxy(props) for cat, props in category_props.items()}

        # TODO: Update for changing mass SMBH
        self.r_g_in_meters = unit_conversion.initialize_r_g(self.smbh_mass)


    def __getattr__(self, item: str) -> Any:
        """
        Retrieves a category proxy (settings.sim) or a flat setting value
        (settings.smbh_mass) from the final compiled dictionary.

        Parameters
        ----------
        item : str
            A category name or setting name.

        Returns
        -------
        Any
            A CategoryProxy for category-level settings, or the final value of the setting.

        Raises
        ------
        AttributeError
            If neither a category nor a setting is found with that name.
        """

        # Compatibility for deepcopy
        if item.startswith("__"):
            raise AttributeError(
                f"attempted to get missing private attribute '{item}'"
            )

        # Category access: settings.sim -> CategoryProxy
        if item in self._categories:
            return self._categories[item]

        # Flat access: settings.smbh_mass (backwards compatibility)
        try:
            return self.settings_finals[item]
        except KeyError:
            raise AttributeError(f"SettingsManager has no key {item!r}")


    def add_custom_category(self, category: str, props: dict[str, Any]) -> None:
        """
        Adds a custom settings category with user-defined properties to the SettingsManager.
        Custom settings are only accessible via settings.category.property_name
        and are not added to the flat settings_finals dictionary.

        Parameters
        ----------
        category : str
            The name of the custom category to create.
        props : dict[str, Any]
            A dictionary of property names and their values.

        Raises
        ------
        ValueError
            If the category name conflicts with an existing category or setting.
        """
        if category in self._categories:
            raise ValueError(f"Category '{category}' already exists")
        if category in self.settings_finals:
            raise ValueError(f"'{category}' conflicts with an existing setting name")

        self._categories[category] = CategoryProxy(props)


    @property
    def categories(self):
        return self._categories


class AGNDisk:
    """
    Container class for the construct_disk_interp method, allowing for pass-by-reference access to disk functions.
    """

    def __init__(self, settings: SettingsManager):
        # TODO: More advanced handling?

        (
            self.disk_surface_density,
            self.disk_aspect_ratio,
            self.disk_opacity,
            self.disk_sound_speed,
            self.disk_density,
            self.disk_pressure_grad,
            self.disk_omega,
            self.disk_surface_density_log,
            self.temp_func,
            self.disk_dlog10surfdens_dlog10R_func,
            self.disk_dlog10temp_dlog10R_func,
            self.disk_dlog10pressure_dlog10R_func
        ) = ReadInputs.construct_disk_interp(
            settings.smbh_mass,
            settings.disk_radius_outer,
            settings.disk_model_name,
            settings.disk_alpha_viscosity,
            settings.disk_bh_eddington_ratio,
            disk_radius_max_pc=settings.disk_radius_max_pc,
            flag_use_pagn=settings.flag_use_pagn,
            verbose= 1 if settings.verbose else 0
        )

        self._settings = settings

    @property
    def settings(self):
        return self._settings
