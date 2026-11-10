"""Define the SettingsManager object and various components"""
import types
######## Imports ########
#### Standard Library ####
import warnings
from types import NoneType
from typing import Any, TypeVar, Type

#### Third Party ####
import numpy as np

#### McFACTS ####
from mcfacts.utilities import unit_conversion

#### Setup ####
IGNORE_ARGS = [
    "print_version",
    "subcommand",
    "enable_profiling",
    "profiling_file",
]

class SettingsProperty:
    T = TypeVar("T")

    def __init__(self, name: str, category: str, value: Any,
                 prop_type: Type[T] = str, aliases: list[str] | None = None):

        # Validate inputs
        if not name or not name.strip():
            raise ValueError("SettingsProperty: name cannot be empty or whitespace")
        if not category or not category.strip():
            raise ValueError("SettingsProperty: category cannot be empty or whitespace")

        # Strict check that the input type matches the expected type
        if type(value) is not prop_type:
            raise TypeError(f"SettingsProperty: the type of value ({type(value)}) must match prop_type {prop_type}")

        # Check that the aliases are an array instead of a single string
        if isinstance(aliases, str):
            raise TypeError("SettingsProperty: aliases must be a list of str, not a str")

        # Check that all of the passed aliases are not empty or whitespace
        aliases = aliases if aliases is not None else []
        for alias in aliases:
            if not alias or not alias.strip():
                raise ValueError("SettingsProperty: aliases cannot be empty or whitespace")

        self._name = name
        self._category = category
        self._value = value
        self._prop_type = prop_type
        self._aliases = tuple(aliases or ())


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
    def aliases(self) -> tuple[str, ...]:
        return self._aliases


    @property
    def prop_type(self) -> Type[T]:
        return self._prop_type


    def __str__(self):
        return f"{self.category}.{self.name}"


    def __eq__(self, other):
        """Check if one SettingsProperty is equal to another SettingsProperty

        Parameters
        ----------
        other : SettingsProperty object
            The alternative property
        """

        if not isinstance(other, self.__class__):
            return NotImplemented

        return (self.name == other.name and
                self.category == other.category and
                self.value == other.value and
                self.prop_type == other.prop_type and
                self.aliases == other.aliases)


class StaticSettingsProperty(SettingsProperty):
    T = TypeVar("T")

    def __init__(self, name: str, category: str, value: Any,
                 prop_type: Type[T] = str, aliases: list[str] | None = None):
        super().__init__(name, category, value, prop_type, aliases)


    @property
    def name(self) -> str:
        return self._name


    @property
    def category(self) -> str:
        return self._category


    @property
    def value(self) -> T:
        return self._value


    @property
    def prop_type(self) -> Type[T]:
        return self._prop_type


    @property
    def aliases(self) -> tuple[str, ...]:
        return self._aliases


class OptionalSettingsProperty(SettingsProperty):
    T = TypeVar("T")

    def __init__(self, name: str, category: str, value: Any,
                 prop_type: Type[T] = str, aliases: list[str] | None = None):
        super().__init__(name, category, value, prop_type, aliases)


DEFAULT_SETTINGS: list[SettingsProperty | StaticSettingsProperty] = [
        # IO Parameters
        SettingsProperty("verbose", "io", False, bool),
        SettingsProperty("show_timeline_progress", "io", False, bool),
        SettingsProperty("overwrite_files", "io", False, bool),
        SettingsProperty("save_state", "io", False, bool),
        SettingsProperty("save_each_timestep", "io", False, bool),
        SettingsProperty("output_dir", "io", "./runs", str),
        SettingsProperty("settings_snapshot", "io", "ini", str),
        SettingsProperty("cabinet_snapshot", "io", "txt", str),
        OptionalSettingsProperty("settings_file", "io", "", str),

        # HDF5 Snapshot Parameters
        SettingsProperty("hdf5_snapshot_file", "hdf5", "model.hdf5", str),
        SettingsProperty("hdf5_snapshot_label", "hdf5", "runs", str),
        SettingsProperty("hdf5_snapshot_mode", "hdf5", "compound", str),
        SettingsProperty("hdf5_snapshot_compression", "hdf5", "gzip", str),
        SettingsProperty("hdf5_snapshot_retries", "hdf5", 3, int),
        SettingsProperty("hdf5_snapshot_sleep", "hdf5", 1., float),

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
        SettingsProperty("mu_circ_encounter", "bh", 0.9, float, aliases=["mean_harden_energy_delta"]),
        SettingsProperty("sigma_circ_encounter", "bh", 0.025, float, aliases=["var_harden_energy_delta"]),
        SettingsProperty("mu_ecc_encounter", "bh", 0.1, float, aliases=["delta_energy_strong_mu"]),
        SettingsProperty("sigma_ecc_encounter", "bh", 0.02, float, aliases=["delta_energy_strong_sigma"]),
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

        # Scaling settings
        SettingsProperty("disk_truncation", "scale", "none", str),
        SettingsProperty("flag_use_scaling", "scale", False, bool),
        SettingsProperty("stellar_mass", "scale", 1e10, float),
        SettingsProperty("scale_smbh_mass", "scale", "schramm-silverman", str),
        SettingsProperty("scale_nsc_mass", "scale", "neumayer-early", str),
        SettingsProperty("scale_inner_disk", "scale", "decay-time", str),
        SettingsProperty("scale_trap", "scale", "sqrt-smbh", str),
        SettingsProperty("scale_capture_radius", "scale", "sqrt-smbh", str),
        SettingsProperty("scale_capture_time", "scale", "hubble", str),

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


DEFAULT_KEYS = {setting.name for setting in DEFAULT_SETTINGS}
DEFAULT_ALIASES_KEYS = {alias for setting in DEFAULT_SETTINGS for alias in setting.aliases}
DEFAULT_CATEGORIES = {setting.category for setting in DEFAULT_SETTINGS}

assert not DEFAULT_KEYS & DEFAULT_ALIASES_KEYS


class CategoryProxy:
    """
    A proxy object that holds properties for a single settings category
    """

    def __init__(self, props: dict[str, Any] | None = None, aliases: dict[str, str] | None = None):
        props = props if props is not None else dict()
        aliases = aliases if aliases is not None else dict()

        # Check if any aliases conflicts with an existing property
        for alias, prop_name in aliases.items():
            if alias in props:
                raise AttributeError(f"Alias '{alias}' (for '{prop_name}') conflicts with an existing property name in this category.")
            if prop_name not in props:
                raise AttributeError(f"Alias '{alias}' points to property name '{prop_name}', which doesn't exist in this category.")
            if alias in self.__dict__ or alias in self.__class__.__dict__:
                raise AttributeError(f"Alias '{alias}' (for '{prop_name}') conflicts with an existing attribute of CategoryProxy.")

        # Check that the alias or prop name does not conflict with existing attributes of the class
        for prop_name, value in props.items():
            if prop_name in self.__dict__ or prop_name in self.__class__.__dict__:
                raise AttributeError(f"Property name '{prop_name}' conflicts with an existing attribute of CategoryProxy.")


        self._props: dict[str, Any] = dict(props)
        self._aliases: dict[str, str] = dict(aliases)


    def __getattr__(self, key: str) -> Any:
        if key.startswith("_"):
            raise AttributeError(f"Unable to access private attribute '{key}'")
        try:
            name_alias = self._aliases.get(key)

            if name_alias is not None:
                return self._props[name_alias]

            return self._props[key]
        except KeyError:
            raise AttributeError(f"No property named '{key}' in this category")


    # Prevent direct assignment to properties -- use SettingsManager.set_preprocessing() instead
    def __setattr__(self, key: str, value: Any) -> None:
        if not key.startswith("_"):
            raise AttributeError(f"Cannot set '{key}' directly. Please use SettingsManager.set_preprocessing()")

        super().__setattr__(key, value)


    @property
    def props(self):
        return types.MappingProxyType(self._props)


    @property
    def aliases(self):
        return types.MappingProxyType(self._aliases)


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
        Attempts to safely cast an override value to the expected type of SettingsProperty.

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

        ## NumPy types ##
        if isinstance(override, np.bool) and expected == bool:
            return bool(override)
        if isinstance(override, np.float64) and expected == float:
            return float(override)
        if isinstance(override, np.int64) and expected == int:
            return int(override)

        if isinstance(prop, OptionalSettingsProperty) and isinstance(override, NoneType):
            return override

        raise TypeError(
            f"Setting '{prop.name}' expects type {expected.__name__}, "
            f"but got {type(override).__name__}"
        )


    def __init__(self, setting_overrides: dict[str, Any] | None = None):
        """
        Initializes the SettingsManager with a dictionary of overrides.

        Parameters
        ----------
        setting_overrides : dict[str, Any]
            A dictionary containing user-provided overrides for specific settings.
            Settings may be referenced by name or by any of their aliases.

        Raises
        ------
        TypeError
            If an override value is not the expected type and cannot be cast to it.
        ValueError
            If a setting is given more than once (by name and/or aliases).
        """
        setting_overrides = setting_overrides if setting_overrides is not None else dict()

        # Raise an error if there are overrides that do not correspond to a default setting
        for key in setting_overrides.keys():
            # If the key is in the ignore list or default list, skip it
            if key in IGNORE_ARGS or key in DEFAULT_KEYS or key in DEFAULT_ALIASES_KEYS:
                continue

            raise ValueError(
                f"Setting dictionary contains {key}, which does not exist as a default setting."
                f"Custom settings must be added through the add_custom_category method."
            )

        category_props: dict[str, dict[str, Any]] = {}
        category_aliases: dict[str, dict[str, str]] = {}

        for prop in DEFAULT_SETTINGS:
            keys = [key for key in (prop.name, *prop.aliases) if key in setting_overrides]

            category_aliases.setdefault(prop.category, {}).update({a: prop.name for a in prop.aliases})

            # If the property does not exist in overrides, use the default value
            if len(keys) == 0:
                category_props.setdefault(prop.category, {})[prop.name] = prop.value
                continue

            if len(keys) > 1:
                raise ValueError(f"Setting {prop} was defined more than once: {keys}")

            override_value = setting_overrides[keys[0]]

            # If the property is static, use the default value
            if isinstance(prop, StaticSettingsProperty):
                if self._cast_override(prop, override_value) != prop.value:
                    warnings.warn(
                        f"Static setting '{prop.name}' cannot be overridden. "
                        f"Using default value: {prop.value}",
                        UserWarning,
                        stacklevel=2,
                    )

                category_props.setdefault(prop.category, {})[prop.name] = prop.value
                continue

            # Cast the override value to the correct type
            category_props.setdefault(prop.category, {})[prop.name] = self._cast_override(prop, override_value)

        # Build the final settings list of main category reference
        self._categories = {cat: CategoryProxy(props, category_aliases.get(cat, {})) for cat, props in category_props.items()}

        self.flat_settings: dict[str, Any] = dict()
        self.flat_aliases: dict[str, str] = dict()
        self._update_flat_settings()

        # TODO: Update for changing mass SMBH
        self.r_g_in_meters = unit_conversion.initialize_r_g(self.sim.smbh_mass)


    def _update_flat_settings(self) -> None:
        """
        Updates the final settings dictionary with the default values stored in the category proxies.
        """

        new_flat_settings: dict[str, Any] = {}
        new_flat_aliases: dict[str, str] = {}

        for name, cat in self._categories.items():
            if name not in DEFAULT_CATEGORIES:
                continue

            for key, value in cat.props.items():
                if key in new_flat_settings:
                    raise ValueError(f"Duplicate setting name '{key}' found in category '{name}'")

                new_flat_settings[key] = value

            for alias_key, point_key in cat.aliases.items():
                if alias_key in new_flat_aliases:
                    raise ValueError(f"Duplicate alias name '{alias_key}' found in category '{name}'")

                if alias_key in new_flat_settings:
                    raise ValueError(f"Alias '{alias_key}' already exists as a setting in category '{name}'")

                new_flat_aliases[alias_key] = point_key

        self.flat_settings = new_flat_settings
        self.flat_aliases = new_flat_aliases

    def __getattr__(self, key: str) -> Any:
        """
        Retrieves a category proxy (settings.sim) or a flat setting value
        (settings.smbh_mass) from the final compiled dictionary.

        Parameters
        ----------
        key : str
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
        if key.startswith("_"):
            raise AttributeError(f"Attempted to get missing private attribute '{key}'")

        # Category access
        if key in self._categories:
            return self._categories[key]

        # Flat access
        if key in self.flat_settings:
            return self.flat_settings.get(key)

        if key in self.flat_aliases:
            actual_key = self.flat_aliases[key]
            return self.flat_settings.get(actual_key)

        raise AttributeError(f"Setting '{key}' does not exist.")


    def set_preprocessing(self, key: str, value: Any):
        """
        Sets the value of a particular setting after the time of creation.

        Parameters
        ----------
        key : str
            The "true name" of a setting, as stored in the dictionary
        value : Any
            A value for that setting
        """
        # Get the original key if an alias was passed
        key = self.flat_aliases.get(key, key)

        # Check if key is a setting
        if key not in self.flat_settings:
            raise ValueError(
                f"No such setting: {key}; "
                f"settings: {list(self.flat_settings.keys())}"
            )

        # Identify the property
        found = None
        for prop in DEFAULT_SETTINGS:
            if prop.name == key:
                found = prop
                break

        # Check if the property was found
        if found is None:
            raise ValueError(
                f"Failed to find setting {key} in DEFAULT_SETTINGS."
            )

        # Check static settings
        if isinstance(found, StaticSettingsProperty):
            raise TypeError(
                f"Modifying a StaticSetting is not allowed, "
                "even in preprocessing!"
            )

        # Check value
        final_value = self._cast_override(found, value)

        # If the property exists, set the value in the appropriate category
        self._categories[found.category]._props[found.name] = final_value

        # TODO: Extend for other dynamic settings in the future
        if found.name == "smbh_mass":
            self.r_g_in_meters = unit_conversion.initialize_r_g(self.sim.smbh_mass)

        self._update_flat_settings()


    def __eq__(self, other):
        """Check if one settings object is equal to another settings object
        
        Parameters
        ----------
        other : SettingsManager object
            The alternative settings
        """
        if not isinstance(other, self.__class__):
            return NotImplemented

        ours = {key: (cat.props, cat.aliases) for key, cat in self._categories.items()}
        theirs = {key: (cat.props, cat.aliases) for key, cat in other._categories.items()}

        return ours == theirs


    def copy(self, replace : dict | None = None):
        """Return a copy of the settings

        Parameters
        ----------
        replace : dict
            Replacement parameters for the copy
        """

        if replace is None:
            replace = {}

        new_default = {
            setting.name: self.flat_settings[setting.name]
            for setting in DEFAULT_SETTINGS if not isinstance(setting, StaticSettingsProperty)
        }

        new_manager = self.__class__(new_default)

        for category, proxy in self._categories.items():
            if category not in new_manager._categories:
                new_manager.add_custom_category(category, proxy.props, aliases=proxy.aliases)

        for key, value in replace.items():
            new_manager.set_preprocessing(key, value)

        return new_manager


    def add_custom_category(self, category: str, props: dict[str, Any], aliases: dict[str, str] | None = None) -> None:
        """
        Adds a custom settings category with user-defined properties to the SettingsManager.
        Custom settings are only accessible via settings.category.property_name
        and are not added to the flat settings dictionary.

        Parameters
        ----------
        category : str
            The name of the custom category to create.
        props : dict[str, Any]
            A dictionary of property names and their values.
        aliases : dict[str, str] | None
            A dictionary mapping aliases to property names within the custom category.

        Raises
        ------
        AttributeError
            If the category name conflicts with an existing category or setting.
        """

        aliases = aliases if aliases is not None else dict()

        current_props = {key for cat in self._categories.values() for key in (*cat.props, *cat.aliases)}

        if category in self._categories:
            raise AttributeError(f"Category '{category}' already exists")

        if category in current_props:
            raise AttributeError(f"'{category}' conflicts with an existing setting name")

        for alias in aliases:
            if alias in current_props:
                raise AttributeError(f"Alias '{alias}' conflicts with an existing setting")

        for prop in props.keys():
            if prop in current_props:
                raise AttributeError(f"Property '{prop}' conflicts with an existing setting")


        self._categories[category] = CategoryProxy(props, aliases)

        # If the update fails for any reason, remove the category we just added.
        try:
            self._update_flat_settings()
        except Exception:
            del self._categories[category]
            raise


    @property
    def categories(self):
        return self._categories

    def new_settings_snapshot(self):
        """Return a new settings snapshot"""
        from mcfacts.objects.snapshot import TxtSnapshotHandler
        from mcfacts.objects.snapshot import IniSnapshotHandler

        from mcfacts.objects.snapshot import HDF5SnapshotHandler
        if self.settings_snapshot == "txt":
            return TxtSnapshotHandler(settings=self) #Neat that this is allowed
        elif self.settings_snapshot == "ini":
            return IniSnapshotHandler(settings=self)
        elif self.settings_snapshot == "hdf5":
            return HDF5SnapshotHandler(settings=self)
        else:
            raise ValueError(
                f"No such snapshot handler: {self.settings_snapshot}",
            )

    def new_cabinet_snapshot(self):
        """Return a new cabinet snapshot"""
        from mcfacts.objects.snapshot import TxtSnapshotHandler
        from mcfacts.objects.snapshot import HDF5SnapshotHandler

        if self.cabinet_snapshot == "txt":
            return TxtSnapshotHandler(settings=self)
        elif self.cabinet_snapshot == "ini":
            raise NotImplementedError(
                f"configparser should not be used to save agn_objects."
            )
        elif self.cabinet_snapshot == "hdf5":
            return HDF5SnapshotHandler(settings=self)
        else:
            raise ValueError(
                f"No such snapshot handler: {self.cabinet_snapshot}",
            )

