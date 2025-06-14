import os
from abc import ABC, abstractmethod
from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from mcfacts.inputs.settings_manager import SettingsManager
from mcfacts.objects.agn_object_array import FilingCabinet, AGNObjectArray
from mcfacts.objects.log import LogFunction


class SnapshotHandler(ABC):
    def __init__(self, name: str, settings: SettingsManager = None):
        self.name = name
        self.settings = settings
        self.parent_log_func: LogFunction = None

    @abstractmethod
    def save_cabinet(self, file_path: str | bytes | PathLike, file_name: str | bytes | PathLike, filing_cabinet: FilingCabinet):
        return NotImplemented

    @abstractmethod
    def load_cabinet(self, file_path: str | bytes | PathLike, file_name: str | bytes | PathLike) -> FilingCabinet:
        return NotImplemented

    @abstractmethod
    def save_settings(self, file_path: str | bytes | PathLike, file_name: str | bytes | PathLike, settings: SettingsManager = None):
        return NotImplemented

    @abstractmethod
    def load_settings(self, file_path: str | bytes | PathLike, file_name: str | bytes | PathLike) -> SettingsManager:
        return NotImplemented

    def set_log_func(self, log_func: LogFunction) -> None:
        self.parent_log_func = log_func

    def log(self, msg: str, new_line: bool = False) -> None:
        if not self.settings.verbose:
            return

        msg = f"{self.name} :: {msg}"

        if self.parent_log_func is None:
            print(f"{(os.linesep if new_line else '')}(ID:??) {msg}")
        else:
            self.parent_log_func(msg, new_line)

    def __str__(self) -> str:
        return f"{self.name} ({type(self)})"


class TxtSnapshotHandler(SnapshotHandler):
    def __init__(self, name: str = None, settings: SettingsManager = None):
        super().__init__("Text Snapshot Handler" if name is None else name, settings)

    @staticmethod
    def get_fully_qualified_type(obj):
        typ = type(obj)

        return typ.__name__ if typ.__module__ == 'builtins' else f"{typ.__module__}.{typ.__name__}"

    def save_cabinet(self, file_path: str | bytes | PathLike, file_name: str | bytes | PathLike, filing_cabinet: FilingCabinet):
        agn_objects: dict[str, AGNObjectArray] = filing_cabinet.agn_objects
        everything_else: dict[str, Any] = filing_cabinet.everything_else

        directory = Path(file_path)
        directory.mkdir(parents=True, exist_ok=True)

        # Handle array objects that exist in the filing cabinet
        for array_name, object_array in agn_objects.items():
            final_path = os.path.join(file_path, file_name + f"_{array_name}.txt")
            super_dict = object_array.get_super_dict()

            keys = super_dict.keys()
            type_array = []

            for key in keys:
                value = super_dict[key]

                if len(value) == 0:
                    type_array.append(f"numpy.{value.dtype}")
                    continue

                type_array.append(self.get_fully_qualified_type(value[0]))

            header = "".join(
                f"{key + '::' + str(type_array[i]) :<{(37 if key.startswith('unique_id') else 33) - (2 if i == 0 else 0)}}"
                for i, key in enumerate(super_dict.keys())
            )

            np.savetxt(final_path, np.column_stack(tuple(super_dict.values())), fmt='%-32s', header=header)

        # Handle the 'everything else' dictionary stored in the filing cabinet
        if len(everything_else) == 0:
            return

        everything_else_path = os.path.join(file_path, file_name + "_everything_else.txt")

        temp_keys = list(everything_else.keys())
        temp_values = list(everything_else.values())
        temp_types = list(self.get_fully_qualified_type(x) for x in everything_else.values())
        temp_array = np.column_stack(tuple([temp_keys, temp_values, temp_types]))

        everything_else_header = "".join(
            f"{key:<{(37 if key.startswith('unique_id') else 26) - (2 if i == 0 else 0)}}"
            for i, key in enumerate(["key", "value", "type"])
        )

        np.savetxt(everything_else_path, temp_array, fmt='%-25s', header=everything_else_header)


    def load_cabinet(self, file_path: str | bytes | PathLike, file_name: str | bytes | PathLike) -> FilingCabinet:
        pass

    def save_settings(self, file_path: str | bytes | PathLike, file_name: str | bytes | PathLike, settings: SettingsManager = None):
        pass

    def load_settings(self, file_path: str | bytes | PathLike, file_name: str | bytes | PathLike) -> SettingsManager:
        pass

