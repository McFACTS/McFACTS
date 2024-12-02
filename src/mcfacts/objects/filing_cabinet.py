import uuid
from typing import TypeVar, Type, Any

from mcfacts.objects.agn_object_array import AGNObjectArray


class FilingCabinet:
    def __init__(self):
        self.agn_objects: dict[str, AGNObjectArray] = dict()
        self.everything_else: dict[str, Any] = dict()

    def list_occurrence(self, unique_id: uuid.UUID) -> list[str]:
        occurrence: list[str] = list()

        for key, value in self.agn_objects.items():
            for entry in value.unique_id:
                if unique_id == entry:
                    occurrence.append(key)

        return occurrence

    def consistency_check(self):
        for value in self.agn_objects.values():
            for entry in value.unique_id:
                occurrence = self.list_occurrence(entry)

                if len(occurrence) > 1:
                    raise RuntimeError(f"A duplicate entry has been found in the filing cabinet. {entry} Found in: {occurrence}")

    def create_or_append_array(self, name: str, agn_object_array: AGNObjectArray):
        if len(agn_object_array) == 0:
            return

        if not isinstance(agn_object_array, AGNObjectArray):
            raise TypeError(f"The filing cabinet does not except type of f{type(agn_object_array).__name__}")

        if name in self.agn_objects.keys():
            self.agn_objects[name].add_objects(agn_object_array)
        else:
            self.set_array(name, agn_object_array)

    def set_array(self, name: str, agn_object_array: AGNObjectArray):
        if not isinstance(agn_object_array, AGNObjectArray):
            raise TypeError(f"The filing cabinet does not except type of f{type(agn_object_array).__name__}")

        self.agn_objects[name] = agn_object_array

    T = TypeVar("T", bound=AGNObjectArray)

    def get_array(self, name: str, agn_object_array_class: Type[T] = AGNObjectArray, create_empty_if_missing=False) -> T:
        if create_empty_if_missing and name not in self.agn_objects.keys():
            empty_agn_object = agn_object_array_class()
            self.set_array(name, empty_agn_object)

        if name not in self.agn_objects.keys():
            raise AttributeError(f"{name} does not exist in the filing cabinet.")

        attribute_array = self.agn_objects[name]

        if not isinstance(attribute_array, agn_object_array_class):
            raise TypeError(f"{name} is not an instance of {agn_object_array_class}")

        return attribute_array

    U = TypeVar("U")

    def get_value(self, name: str, default_value: U = None) -> U:
        if default_value is not None:
            self.set_value(name, default_value)

        if name not in self.everything_else.keys():
            raise AttributeError(f"{name} does not exist in the filing cabinet.")

        value = self.everything_else[name]

        return value

    def set_value(self, name: str, value: Any):
        self.everything_else[name] = value

    def __contains__(self, item):
        return item in self.agn_objects.keys() or item in self.everything_else.keys()

    def __len__(self):
        return len(self.agn_objects) + len(self.everything_else)
