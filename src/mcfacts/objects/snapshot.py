"""Module for the serialization of McFACTS simulation data"""
######## Imports ########
#### Standard Library ####
import configparser
import os
import uuid
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Any

#### Third Party ####
import numpy as np
import pandas as pd

#### Vera ####
from xdata import Database

#### McFACTS ####
from mcfacts.inputs import settings_manager
from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.objects.agn_object_array import FilingCabinet, AGNObjectArray
from mcfacts.objects.log import LogFunction, PrintLogFunction

######## Setup ########

UUID_FIELDS = [
    "unique_id",
    "parent_unique_id",
    "parent_unique_id_2",
    "progenitor_unique_id",
    "unique_id_final",
]

######## Objects ########
class SnapshotHandler(ABC):
    def __init__(self, name: str, settings: SettingsManager = None):
        self.name = name
        self.settings = settings
        self.parent_log_func: LogFunction = PrintLogFunction(
            prefix=f"(ID:??) {self.name} :: "
        )

    @abstractmethod
    def save_cabinet(
            self,
            directory       : str | bytes | PathLike,
            file_name       : str | bytes | PathLike,
            filing_cabinet  : FilingCabinet,
        ):
        return NotImplemented

    @abstractmethod
    def load_cabinet(
            self,
            directory: str | bytes | PathLike,
            file_name: str | bytes | PathLike,
        ) -> Any:
        return NotImplemented

    @abstractmethod
    def save_settings(
            self,
            directory: str | bytes | PathLike,
            file_name: str | bytes | PathLike,
            settings: SettingsManager = None,
        ):
        return NotImplemented

    @abstractmethod
    def load_settings(
            self,
            directory: str | bytes | PathLike,
            file_name: str | bytes | PathLike,
        ) -> SettingsManager:
        return NotImplemented

    def set_log_func(self, log_func: LogFunction) -> None:
        if not isinstance(log_func, LogFunction):
            raise TypeError(
                f"log_func is type {type(log_func)}. "
                f"Should be subclass of {LogFunction}."
            )
        self.parent_log_func = log_func

    def log(self, msg: str, new_line: bool = False) -> None:
        if not self.settings.verbose:
            return
        if new_line:
            self.parent_log_func.new_line()
        self.parent_log_func(msg)

    def __str__(self) -> str:
        return f"{self.name} ({type(self)})"


class TxtSnapshotHandler(SnapshotHandler):
    def __init__(self, name: str = None, settings: SettingsManager = None):
        super().__init__("Text Snapshot Handler" if name is None else name, settings)

    @staticmethod
    def get_fully_qualified_type(obj):
        typ = type(obj)

        return typ.__name__ if typ.__module__ == 'builtins' else f"{typ.__module__}.{typ.__name__}"

    def save_cabinet(
            self,
            directory: str | bytes | PathLike,
            file_name: str | bytes | PathLike,
            filing_cabinet: FilingCabinet,
        ):
        agn_objects: dict[str, AGNObjectArray] = filing_cabinet.agn_objects
        everything_else: dict[str, Any] = filing_cabinet.everything_else

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Handle array objects that exist in the filing cabinet
        for array_name, object_array in agn_objects.items():
            final_path = os.path.join(directory, file_name + f"_{array_name}.txt")
            super_dict = object_array.get_super_dict()

            keys = super_dict.keys()
            type_array = []
            spacing_array = []

            for key in keys:
                values = super_dict[key]

                type_str = f"numpy.{values.dtype}" if len(values) == 0 else f"{self.get_fully_qualified_type(values[0])}"
                final_type_str = f"{key}::{type_str}"
                type_array.append(final_type_str)

                if len(values) == 0:
                    spacing_array.append(len(final_type_str))
                else:
                    longest = max([str(x) for x in values], key=len)
                    spacing_array.append(max(len(longest), len(final_type_str)) + 1)

            header = "".join(
                f"{str(type_array[i]) :<{spacing_array[i]}}" for i, key in enumerate(super_dict.keys())
            )

            np.savetxt(final_path, np.column_stack(tuple(super_dict.values())), fmt=[f"%-{space - 1}s" for space in spacing_array], header=header, comments='')

        # Handle the 'everything else' dictionary stored in the filing cabinet
        if len(everything_else) == 0:
            return

        everything_else_path = os.path.join(directory, file_name + "_everything_else.txt")

        temp_keys = list(everything_else.keys())
        temp_values = list(everything_else.values())
        temp_types = list(self.get_fully_qualified_type(x) for x in everything_else.values())
        temp_array = np.column_stack(tuple([temp_keys, temp_values, temp_types]))

        everything_else_header = "".join(
            f"{key:<{(37 if key.startswith('unique_id') else 26)}}"
            for i, key in enumerate(["key", "value", "type"])
        )

        np.savetxt(everything_else_path, temp_array, fmt='%-25s', header=everything_else_header, comments='')


    def load_cabinet(
            self,
            directory: str | bytes | PathLike,
            file_name: str | bytes | PathLike,
        ) -> dict:
        directory = Path(directory)

        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        agn_objects = dict()
        everything_else = dict() # TODO: Handle everything else dictionary

        for file in directory.iterdir():
            if not file.is_file():
                continue
            if not file.name.startswith(f"{file_name}_"):
                continue
            if not file.name.lower().endswith(".txt"):
                continue
            if "everything_else" in str(file):
                with open(file, 'r') as F:
                    for line in F:
                        if line.startswith("key"):
                            continue
                        line = line.strip("\n")
                        parts = line.split()
                        if len(parts) != 3:
                            continue
                        key, value, dt = parts
                        # This is bad. Do not do this.
                        #value = eval(dt)(value)
                        if dt == "float":
                            value = float(value)
                        else:
                            raise TypeError(
                                f"Bad type in everything_else: {dt}"
                            )
                        everything_else[key] = value
                continue

            array_name = file.name[len(file_name + "_"):].removesuffix(".txt")

            if not array_name:
                continue

            try:
                data = pd.read_csv(file, sep=r"\s+", dtype=str, engine='python')
            except Exception as ex:
                print(f"Failed to load {file}: {ex}")
                continue

            if len(data) == 0:
                # VD: NOTE this continue here is not good Python.
                # It should return an empty array.
                # This is some pandas nonsense, and I'm not fixing it.
                continue

            column_dict = dict()

            for series_name, series in data.items():
                key, value = str(series_name).split('::')

                if value == "uuid.UUID":
                    array = np.array([uuid.UUID(v) for v in series], dtype=uuid.UUID)
                elif value.startswith("numpy."):
                    array = np.array(series, np.dtype(value.split('.')[1]))
                else:
                    array = np.array(series)

                column_dict[key] = array

            agn_objects[array_name] = column_dict

        return agn_objects, everything_else


    def save_settings(
            self,
            directory: str | bytes | PathLike,
            file_name: str | bytes | PathLike,
            settings: SettingsManager = None,
        ):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if not file_name.lower().endswith(".txt"):
            file_name = file_name + ".txt"
        final_path = os.path.join(directory, file_name)

        temp_keys = list(settings.settings_finals.keys())
        temp_values = list(settings.settings_finals.values())
        temp_types = list(self.get_fully_qualified_type(x) for x in settings.settings_finals.values())

        # Fixes a bug with an unset "settings_file" that I found in testing
        for i, (t, key, val) in enumerate(zip(temp_types, temp_keys, temp_values)):
            if isinstance(val, str) and len(val) == 0:
                if key == "settings_file":
                    temp_values[i] = os.path.basename(file_name)
                else:
                    raise ValueError(
                        f"setting {key} had value {val} with length 0. "
                        "TxtSnapshotHandler cannot preserve empty strings."
                    )
                
        # Join arrays
        temp_array = np.column_stack(tuple([temp_keys, temp_values, temp_types]))

        spacing_array = []

        longest = max([str(x) for x in temp_keys], key=len)
        spacing_array.append(len(longest) + 1)

        longest = max([str(x) for x in temp_values], key=len)
        spacing_array.append(len(longest) + 1)

        longest = max([str(x) for x in temp_types], key=len)
        spacing_array.append(len(longest) + 1)

        settings_header = "".join(
            f"{key:<{spacing_array[i]}}"
            for i, key in enumerate(["key", "value", "type"])
        )

        np.savetxt(final_path, temp_array, fmt=[f"%-{space - 1}s" for space in spacing_array], header=settings_header, comments='')


    def load_settings(
            self,
            directory: str | bytes | PathLike,
            file_name: str | bytes | PathLike,
        ) -> SettingsManager:
        if not file_name.lower().endswith(".txt"):
            file_name = file_name + ".txt"
        final_path = os.path.join(directory, file_name)

        # Loade the settings from the txt snapshot
        data = np.genfromtxt(final_path, skip_header=1, dtype=str)

        settings = {}

        for row in data:
            name = str(row[0])
            value = str(row[1])
            type_str = str(row[2])

            if type_str == "int":
                settings[name] = int(value)
            elif type_str == "float":
                settings[name] = float(value)
            elif type_str == "str":
                settings[name] = str(value)
            elif type_str == "bool":
                settings[name] = value  # Let SettingsManager._cast_override handle bool parsing
            else:
                raise TypeError(f"Unknown type '{type_str}' for setting '{name}'")

        return SettingsManager(settings)


class IniSnapshotHandler(SnapshotHandler):
    def __init__(self, name: str = None, settings: SettingsManager = None):
        super().__init__("Ini Snapshot Handler" if name is None else name, settings)

    def save_cabinet(
            self,
            directory: str | bytes | PathLike,
            file_name: str | bytes | PathLike,
            filing_cabinet: FilingCabinet,
        ):
        raise NotImplementedError("IniSnapshotHandler does not support saving FilingCabinets")

    def load_cabinet(
            self,
            directory: str | bytes | PathLike,
            file_name: str | bytes | PathLike,
        ) -> Any:
        raise NotImplementedError("IniSnapshotHandler does not support loading FilingCabinets")

    def save_settings(
            self,
            directory: str | bytes | PathLike,
            file_name: str | bytes | PathLike,
            settings: SettingsManager = None,
        ):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        if not file_name.lower().endswith(".ini"):
            file_name = file_name + ".ini"
        final_path = os.path.join(directory, file_name)

        name_to_category = {prop.name: prop.category for prop in settings_manager.DEFAULT_SETTINGS}

        config = configparser.ConfigParser()

        # Write all standard settings, grouped by category
        for name, value in settings.settings_finals.items():
            category = name_to_category.get(name, "custom")
            if not config.has_section(category):
                config.add_section(category)
            config.set(category, name, str(value))

        # Write any custom categories that aren't in settings_finals
        standard_categories = {prop.category for prop in settings_manager.DEFAULT_SETTINGS}
        for category, proxy in settings.categories.items():
            if category in standard_categories:
                continue
            if not config.has_section(category):
                config.add_section(category)
            for name, value in proxy._props.items():
                config.set(category, name, str(value))

        with open(final_path, "w") as f:
            config.write(f)

        self.log(f"Saved settings to {final_path}")

    def load_settings(
            self,
            directory: str | bytes | PathLike,
            file_name: str | bytes | PathLike,
        ) -> SettingsManager:
        if not file_name.lower().endswith(".ini"):
            file_name = file_name + ".ini"
        final_path = os.path.join(directory, file_name)

        if not Path(final_path).exists():
            raise FileNotFoundError(f"Settings file not found: {final_path}")

        config = configparser.ConfigParser()
        config.read(final_path)

        standard_setting_names = {prop.name for prop in settings_manager.DEFAULT_SETTINGS}
        standard_categories = {prop.category for prop in settings_manager.DEFAULT_SETTINGS}

        settings_overrides = {}
        custom_categories = {}

        for section in config.sections():
            if section in standard_categories:
                # Treat standard sections normally, by passing values into init fields
                for name, value in config[section].items():
                    if name in standard_setting_names:
                        settings_overrides[name] = value
            else:
                # Save custom section for later to be loaded with method
                custom_categories[section] = dict(config[section].items())

        print(settings_overrides)
        manager = SettingsManager(settings_overrides)

        for category, props in custom_categories.items():
            manager.add_custom_category(category, props)

        return manager

class HDF5SnapshotHandler(SnapshotHandler):
    def __init__(
            self,
            name    : str = None,
            settings: SettingsManager = None,
            addr    : str = None,
        ):
        super().__init__("HDF5 Snapshot Handler" if name is None else name, settings)
        self.addr = addr

    @property
    def needle(self):
        """Setting to search for to find settings

        Name a setting that is unlikely to have a run named after it,
          which is also unlikely to have its name be changed.

        A needle in a haystack
        """
        return "disk_radius_outer"

    @staticmethod
    def construct_path(directory, file_name):
        # Assert the directory is a path
        if not isinstance(directory, Path):
            raise TypeError(
                f"directory is type {type(directory)}; expected Path"
            )
        # Make file_name a Path
        file_name = Path(file_name)
        suffix = file_name.suffix
        if suffix.lower() in [".hdf5", ".hdf", ".h5"]:
            pass
        elif len(suffix.lower()) == 0:
            file_name = Path(str(file_name) + ".hdf5")
        else:
            raise ValueError(f"Unknown file extension: {suffix}")
        return directory / file_name

    @property
    def label(self):
        if self.addr is not None:
            return self.addr
        elif (self.settings is None) or \
                (self.settings.settings_file is None) or \
                (len(self.settings.settings_file) == 0):
            return "runs"
        else:
            return Path(self.settings.settings_file).stem

    @property
    def settings_addr(self):
        if self.addr is not None:
            return self.addr # Note there is no /settings
        else:
            return self.label + "/settings"

    def save_cabinet(
            self,
            directory       : str | bytes | PathLike,
            file_name       : str | bytes | PathLike,
            filing_cabinet  : FilingCabinet,
            addr            : str = None,
        ):
        """Save a cabinet to HDF5 using xdata

        Parameters
        ----------
        directory   : path_like
            The location of the directory where the file will be saved
        file_name   : path_like
            The name of the file that will be saved
        filing_cabinet : FilingCabinet
            The cabinet with your population
        addr        : str, optional
            The address within the hdf5 file to save settings
        """
        agn_objects: dict[str, AGNObjectArray] = filing_cabinet.agn_objects
        everything_else: dict[str, Any] = filing_cabinet.everything_else

        # Create the directory if it does not already exist
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # Make file_name a Path
        final_path = self.construct_path(directory, file_name)
        # Open the database
        if addr is None:
            if self.addr is not None:
                addr = self.addr
            else:
                addr = self.label + "/population"
        db = Database(final_path, addr)

        # Handle array objects that exist in the filing cabinet
        for array_name, object_array in agn_objects.items():
            # Up, up, and away!
            super_dict = object_array.get_super_dict()
            if array_name not in db.list_items(kind='group'):
                # path relative to addr
                db.create_group(array_name)
            for key, value in super_dict.items():
                # path relative to addr
                tag = f"{array_name}/{key}"
                if key in UUID_FIELDS:
                    value = np.array([u.bytes for u in value], dtype='S16')
                try:
                    db.dset_set(tag, value)
                except Exception as exc:
                    print(key, value.dtype, np.shape(value))
                    raise exc

        # If there's nothing else, we're done here
        if len(everything_else) == 0:
            return
        # Handle the 'everything else' dictionary stored in the filing cabinet
        for key, value in everything_else.items():
            # path relative to addr
            db.dset_set(key, np.asarray(value))


    def load_cabinet(
            self,
            directory   : str | bytes | PathLike,
            file_name   : str | bytes | PathLike,
            addr        : str = None,
        ) -> dict:
        """Load a cabinet from HDF5 using xdata

        Parameters
        ----------
        directory   : path_like
            The location of the directory where the file exists
        file_name   : path_like
            The name of the file that will be loaded
        addr        : str, optional
            The address within the hdf5 file to load a cabinet
        """
        # Create the directory Path
        directory = Path(directory)
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        # Make file_name a Path
        final_path = self.construct_path(directory, file_name)
        if not final_path.exists():
            raise FileNotFoundError(f"File not found: {final_path}")
        # Open the database
        db = Database(final_path)
        # Easy: the user told us where the cabinet is
        # Note this is the only way to save a cabinet for a galaxy
        if addr is not None:
            if not db.exists(addr):
                db.create_group(addr)
        # Hard: let's try and look for them
        else:
            top = db.list_items()
            if "population" in top:
                addr = "population"
            elif len(top) == 1:
                for item in db.list_items(top[0]):
                    if item == "population":
                        addr = f"{item}/population"
        # Make sure we found something
        if addr is None:
            raise RuntimeError(f"Please specify HDF5 cabinet address!")
        # Point database
        db = Database(final_path, addr)

        # The dictionary
        agn_objects = dict()
        everything_else = dict() # TODO: Handle everything else dictionary

        # First load AGN objects
        for item in db.list_items(kind="group"):
            kind = db.kind(item)
            # Initialize column dict
            column_dict = dict()
            for key in db.list_items(item, kind="dset"):
                if key in UUID_FIELDS:
                    # Load the bytes array into RAM
                    tmp = db.dset_value(f"{item}/{key}")
                    col = np.empty(tmp.shape, dtype=object)
                    for i, bts in enumerate(tmp):
                        lbts = len(bts)
                        if lbts == 0:
                            col[i] = uuid.UUID(int=0)
                        elif lbts == 16:
                            col[i] = uuid.UUID(bytes=bts)
                        elif lbts < 16:
                            col[i] = uuid.UUID(bytes=bts.ljust(16, b"\x00"))
                        else:
                            raise ValueError(
                                f"UUID bytearray has an element with {lbts} bytes!"
                            )
                    # Initialize column_dict
                    column_dict[key] = col
                else:
                    # EZ
                    column_dict[key] = db.dset_value(f"{item}/{key}")
            agn_objects[item] = column_dict

        # Handle everything else
        for item in db.list_items(kind="dset"):
            everything_else[item] = db.dset_value(item)

        return agn_objects, everything_else


    def save_settings(
            self,
            directory   : str | bytes | PathLike,
            file_name   : str | bytes | PathLike,
            settings    : SettingsManager = None,
            addr        : str = None,
        ):
        """Save settings to HDF5 using xdata

        Parameters
        ----------
        directory   : path_like
            The location of the directory where the file will be saved
        file_name   : path_like
            The name of the file that will be saved
        settings    : SettingsManager, optional
            The settings to save
        addr        : str, optional
            The address within the hdf5 file to save settings
        """
        # Create the directory if it does not already exist
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        # Make file_name a Path
        final_path = self.construct_path(directory, file_name)
        # Determine the address to save the settings within the HDF5 database
        if addr is None:
            addr = self.settings_addr
        # Open the database
        db = Database(final_path, addr)

        ## Save requires mutating data ##
        save_attrs = {key:value for key, value in settings.settings_finals.items()}
        # Cannot save objects to hdf5
        if save_attrs["settings_file"] is None:
            save_attrs["settings_file"] = ""
        # Seed is a u128, 
        #  but NumPy doesn't support a u128 datatype so I don't either
        save_attrs["seed"] = str(save_attrs["seed"])
        # Loop settings (leave this here for testing)
        #for key, value in save_attrs.items():
        #    print(key, type(value), value)
        #    db.attr_set(".", key, value)
        # Save the dictionary
        db.attr_set_dict(".", save_attrs)


    def load_settings(
            self,
            directory   : str | bytes | PathLike,
            file_name   : str | bytes | PathLike,
            addr        : str = None,
        ) -> SettingsManager:
        """Load settings from HDF5 using xdata

        Parameters
        ----------
        directory   : path_like
            The location of the directory where the file is
        file_name   : path_like
            The name of the file
        addr        : str, optional
            The address within the hdf5 file to find the settings

        Returns
        -------
        SettingsManager
            The loaded settings
        """
        # Create the directory Path
        directory = Path(directory)
        # Make file_name a Path
        final_path = self.construct_path(directory, file_name)
        if not final_path.exists():
            raise FileNotFoundError(f"File not found: {final_path}")
        # Open the database
        db = Database(final_path)
        # Initialize settings
        settings = None
        # Easy: the user told us where the settings are
        if addr is not None:
            settings = db.attr_dict(addr)
        # Hard: let's try and look for them
        else:
            top = db.list_items()
            top_attrs = db.attr_dict(".")
            # See if it's right here
            if self.needle in top_attrs:
                settings = top_attrs
            # See if there's something called settings
            elif "settings" in top:
                settings = db.attr_dict("settings")
            # Let's see if there's exactly one group
            elif len(top) == 1:
                # Let's see if this thing has 'settings'
                if "settings" in db.list_items(top[0]):
                    settings = db.attr_dict(f"{top[0]}/settings")
                # Let's see if it has settings
                else:
                    label_attrs = db.attr_dict(top[0])
                    if self.needle in label_attrs:
                        settings = label_attrs
        # Die if we failed to find them
        if settings is None:
            raise KeyError(f"I could not find your settings.")
        # Fix seed
        settings["seed"] = int(settings["seed"])
        return SettingsManager(settings)
