import uuid
from abc import ABC, abstractmethod
from copy import deepcopy
from typing import Any, TypeVar, Type

import numpy as np
import numpy.typing as npt
from typing_extensions import override


class AGNObjectArray(ABC):
    """
    AGNObjectArray is an abstract base class for managing arrays of AGN (Active Galactic Nuclei) objects with specific properties.

    Attributes:
        unique_id (npt.NDArray[uuid]): Array of unique identifiers for each AGN object.
        mass (npt.NDArray[np.float_]): Array of masses for the AGN objects.
        spin (npt.NDArray[np.float_]): Array of spin magnitudes for the AGN objects.
        spin_angle (npt.NDArray[np.float_]): Array of spin angles for the AGN objects.
        orb_a (npt.NDArray[np.float_]): Array of orbital semi-major axes.
        orb_inc (npt.NDArray[np.float_]): Array of orbital inclinations.
        orb_ecc (npt.NDArray[np.float_]): Array of orbital eccentricities.
        orb_ang_mom (npt.NDArray[np.float_]): Array of orbital angular momenta.
        orb_arg_periapse (npt.NDArray[np.float_]): Array of arguments of periapsis.

    Methods:
        __len__(): Returns the number of AGN objects in the array.
        add_objects(agnObjectArray: 'AGNObjectArray'): Abstract method to concatenate properties of another AGNObjectArray into this one. Must be implemented by a subclass.
    """

    def __init__(self,
                 unique_id: npt.NDArray[uuid] = np.array([]),
                 mass: npt.NDArray[np.float_] = np.array([]),
                 spin: npt.NDArray[np.float_] = np.array([]),
                 spin_angle: npt.NDArray[np.float_] = np.array([]),
                 orb_a: npt.NDArray[np.float_] = np.array([]),
                 orb_inc: npt.NDArray[np.float_] = np.array([]),
                 orb_ecc: npt.NDArray[np.float_] = np.array([]),
                 orb_ang_mom: npt.NDArray[np.float_] = np.array([]),
                 orb_arg_periapse: npt.NDArray[np.float_] = np.array([]),
                 gen: npt.NDArray[np.int_] = np.array([])):

        if len(mass) > 0 and len(unique_id) == 0:
            raise ValueError("AGNObjectArray must be initialized with unique_id filled.")

        # Size of id is 16 bytes. In 1 GB of memory, you can store 62,500,000 ids.
        # Since its not raw byte, that number is a bit smaller, but it should not cause any space issues.
        self.unique_id = unique_id

        self.mass = mass
        self.spin = spin
        self.spin_angle = spin_angle
        self.orb_a = orb_a
        self.orb_inc = orb_inc
        self.orb_ecc = orb_ecc
        self.orb_ang_mom = orb_ang_mom
        self.orb_arg_periapse = orb_arg_periapse

        self.gen: npt.NDArray[np.int_] = np.full(len(unique_id), 1) if len(gen) == 0 else gen

        self.consistency_check()

    # Legacy reference to id_num
    @property
    def id_num(self):
        return self.unique_id

    @id_num.setter
    def id_num(self, new_value):
        self.unique_id = new_value

    # Legacy reference to num
    @property
    def num(self):
        return len(self.unique_id)

    def consistency_check(self):
        """
        Verifies the consistency of object attributes by comparing the lengths of arrays in the superclass to the length of the unique ID list.

        This method checks if the length of the `unique_id` list matches the lengths of the corresponding arrays in the superclass.
        If any inconsistency is found, it raises an exception indicating whether an object ID or attribute is missing.

        Raises:
            Exception: If the lengths of the `unique_id` list and any attribute array do not match, an exception is raised.
                      The error message specifies whether the object ID or attribute is missing and the expected vs found lengths.
        """
        id_len = len(self.unique_id)

        for name, attribute in self.get_super_list().items():
            list_length = len(attribute)

            if list_length != id_len:
                if id_len < list_length:
                    raise RuntimeError(f"Missing object id entries of {type(self).__name__} in {name} array. Expected: {list_length}, Found: {id_len}")
                else:
                    raise RuntimeError(f"Missing attribute entries of {type(self).__name__} in {name} array. Expected: {id_len}, Found: {list_length}")

    # Legacy method for at_id_num()
    def at_id_num(self, unique_id: npt.NDArray[uuid.UUID], attribute_name: str):
        return self.get_attribute(attribute_name, unique_id)

    def get_attribute(self, attribute_name: str, unique_id: npt.NDArray[uuid.UUID]) -> npt.NDArray[Any]:
        """
        Retrieves a copy of an array of attribute values corresponding to a given attribute name and a selection of unique IDs.

        Args:
            attribute_name (str): The name of the attribute to retrieve.
            unique_id (npt.ArrayLike[uuid.UUID]): A list or array of unique IDs to filter the attribute values.

        Returns:
            npt.NDArray[Any]: An array containing the values of the specified attribute for the given unique IDs.

        Raises:
            AttributeError: If the specified attribute name does not exist in the superclass attribute list.

        Notes:
            - The method uses a selection mask to filter the attribute array based on the provided unique IDs.
            - Assumes that the `unique_id` attribute of the class contains a list of IDs matching those in the superclass.
        """
        super_list = self.get_super_list()

        if attribute_name not in super_list.keys():
            raise AttributeError(f"{attribute_name} is not an attribute of {type(self).__name__}.")

        selection_mask = np.isin(self.unique_id, unique_id)

        return super_list[attribute_name][selection_mask]

    def remove_all(self, unique_id: npt.NDArray[uuid.UUID]) -> bool:
        """
        Removes all entries from the object's attributes that match the given unique IDs.

        Args:
            unique_id (npt.ArrayLike[uuid.UUID]): A list or array of unique IDs to remove.

        Returns:
            bool: True if any entries were removed; False otherwise.

        Notes:
            - Uses a mask to exclude entries matching the specified `unique_id` values.
            - Updates all attributes in the superclass attribute list to reflect the removed entries.
            - If no entries match the given unique IDs, no changes are made, and the method returns `False`.
        """
        remove_mask = ~np.isin(self.unique_id, unique_id)

        if len(remove_mask) == 0:
            return False

        for attribute_name, attribute_value in self.get_super_list().items():
            setattr(self, attribute_name, attribute_value[remove_mask])

        self.consistency_check()

        return True

    def keep_only(self, unique_id: npt.NDArray[uuid.UUID]) -> bool:
        """
        Keeps only the entries in the object's attributes that match the given unique IDs.

        Args:
            unique_id (npt.ArrayLike[uuid.UUID]): A list or array of unique IDs to retain.

        Returns:
            bool: True if any entries were retained; False otherwise.

        Notes:
            - Uses a mask to include only entries matching the specified `unique_id` values.
            - Updates all attributes in the superclass attribute list to reflect the retained entries.
            - If no entries match the given unique IDs, no changes are made, and the method returns `False`.
        """
        remove_mask = np.isin(self.unique_id, unique_id)

        if len(remove_mask) == 0:
            return False

        for attribute_name, attribute_value in self.get_super_list().items():
            setattr(self, attribute_name, attribute_value[remove_mask])

        self.consistency_check()

        return True

    def copy(self):
        return deepcopy(self)

    @abstractmethod
    def get_super_list(self) -> dict[str, npt.NDArray[Any]]:
        return {
            "unique_id": self.unique_id,
            "mass": self.mass,
            "spin": self.spin,
            "spin_angle": self.spin_angle,
            "orb_a": self.orb_a,
            "orb_inc": self.orb_inc,
            "orb_ecc": self.orb_ecc,
            "orb_ang_mom": self.orb_ang_mom,
            "orb_arg_periapse": self.orb_arg_periapse,
            "gen": self.gen
        }

    @abstractmethod
    def add_objects(self, agnObjectArray: 'AGNObjectArray'):
        if not isinstance(agnObjectArray, AGNObjectArray):
            raise Exception(f"Type Error: Unable to add {type(agnObjectArray)} objects to AGNObjectArray.")

        self.unique_id = np.concatenate((self.unique_id, agnObjectArray.unique_id))
        self.gen = np.concatenate((self.gen, agnObjectArray.gen))
        self.mass = np.concatenate((self.mass, agnObjectArray.mass))
        self.spin = np.concatenate((self.spin, agnObjectArray.spin))
        self.spin_angle = np.concatenate((self.spin_angle, agnObjectArray.spin_angle))
        self.orb_a = np.concatenate((self.orb_a, agnObjectArray.orb_a))
        self.orb_inc = np.concatenate((self.orb_inc, agnObjectArray.orb_inc))
        self.orb_ecc = np.concatenate((self.orb_ecc, agnObjectArray.orb_ecc))
        self.orb_ang_mom = np.concatenate((self.orb_ang_mom, agnObjectArray.orb_ang_mom))
        self.orb_arg_periapse = np.concatenate((self.orb_arg_periapse, agnObjectArray.orb_arg_periapse))

    def __len__(self):
        return len(self.unique_id)


class AGNBlackHoleArray(AGNObjectArray):
    def __init__(self,
                 gw_freq: npt.NDArray[np.float_] = np.array([]),
                 gw_strain: npt.NDArray[np.float_] = np.array([]),
                 **kwargs):

        self.gw_freq: npt.NDArray[np.float_] = gw_freq if len(gw_freq) > 0 else np.full(len(kwargs.get("unique_id")), -1)
        self.gw_strain: npt.NDArray[np.float_] = gw_strain if len(gw_freq) > 0 else np.full(len(kwargs.get("unique_id")), -1)

        # Call init last so consistency check passes.
        super().__init__(**kwargs)

    @override
    def get_super_list(self) -> dict[str, npt.NDArray[Any]]:
        super_list = super().get_super_list()

        super_list["gw_freq"] = self.gw_freq
        super_list["gw_strain"] = self.gw_strain

        return super_list

    @override
    def add_objects(self, agnObjectArray: 'AGNBlackHoleArray'):
        super().add_objects(agnObjectArray)

        if not isinstance(agnObjectArray, AGNBlackHoleArray):
            raise Exception(f"Type Error: Unable to add {type(agnObjectArray)} objects to AGNBlackHoleArray.")

        self.gw_freq = np.concatenate((self.gw_freq, agnObjectArray.gw_freq))
        self.gw_strain = np.concatenate((self.gw_strain, agnObjectArray.gw_strain))


class AGNStarArray(AGNObjectArray):
    def __init__(self,
                 star_x: npt.NDArray[np.float_] = np.array([]),
                 star_y: npt.NDArray[np.float_] = np.array([]),
                 star_z: npt.NDArray[np.float_] = np.array([]),
                 radius: npt.NDArray[np.float_] = np.array([]),
                 **kwargs):
        self.star_x = star_x
        self.star_y = star_y
        self.star_z = star_z,
        self.radius = radius

        # Call init last so consistency check passes.
        super().__init__(**kwargs)

    @override
    def get_super_list(self) -> dict[str, npt.NDArray[Any]]:
        super_list = super().get_super_list()

        super_list["star_x"] = self.star_x
        super_list["star_y"] = self.star_y
        super_list["star_z"] = self.star_z
        super_list["radius"] = self.radius

        return super_list

    @override
    def add_objects(self, agnObjectArray: 'AGNStarArray'):
        super().add_objects(agnObjectArray)

        if not isinstance(agnObjectArray, AGNStarArray):
            raise Exception(f"Type Error: Unable to add {type(agnObjectArray)} objects to AGNStarArray.")

        self.star_x = np.concatenate((self.star_x, agnObjectArray.star_x))
        self.star_y = np.concatenate((self.star_y, agnObjectArray.star_y))
        self.star_z = np.concatenate((self.star_z, agnObjectArray.star_z))
        self.radius = np.concatenate((self.radius, agnObjectArray.radius))


class AGNBinaryBlackHoleArray(AGNBlackHoleArray):
    def __init__(self,
                 unique_id_1: npt.NDArray[uuid] = np.array([]),
                 unique_id_2: npt.NDArray[uuid] = np.array([]),
                 mass_1: npt.NDArray[np.float_] = np.array([]),
                 mass_2: npt.NDArray[np.float_] = np.array([]),
                 orb_a_1: npt.NDArray[np.float_] = np.array([]),
                 orb_a_2: npt.NDArray[np.float_] = np.array([]),
                 spin_1: npt.NDArray[np.float_] = np.array([]),
                 spin_2: npt.NDArray[np.float_] = np.array([]),
                 spin_angle_1: npt.NDArray[np.float_] = np.array([]),
                 spin_angle_2: npt.NDArray[np.float_] = np.array([]),
                 bin_sep: npt.NDArray[np.float_] = np.array([]),
                 bin_orb_a: npt.NDArray[np.float_] = np.array([]),
                 time_to_merger_gw: npt.NDArray[np.float_] = np.array([]),
                 flag_merging: npt.NDArray[np.float_] = np.array([]),
                 time_merged: npt.NDArray[np.float_] = np.array([]),
                 bin_ecc: npt.NDArray[np.float_] = np.array([]),
                 gen_1: npt.NDArray[np.float_] = np.array([]),
                 gen_2: npt.NDArray[np.float_] = np.array([]),
                 bin_orb_ang_mom: npt.NDArray[np.float_] = np.array([]),
                 bin_orb_inc: npt.NDArray[np.float_] = np.array([]),
                 bin_orb_ecc: npt.NDArray[np.float_] = np.array([]),
                 **kwargs
                 ):
        self.unique_id_2 = unique_id_2
        self.mass_2 = mass_2
        self.orb_a_2 = orb_a_2
        self.spin_2 = spin_2
        self.spin_angle_2 = spin_angle_2
        self.bin_sep = bin_sep
        self.bin_orb_a= bin_orb_a
        self.time_to_merger_gw = time_to_merger_gw
        self.flag_merging = flag_merging
        self.time_merged = time_merged
        self.bin_ecc = bin_ecc
        self.gen_2 = gen_2
        self.bin_orb_ang_mom = bin_orb_ang_mom
        self.bin_orb_inc = bin_orb_inc
        self.bin_orb_ecc = bin_orb_ecc

        # Call init last so consistency check passes.
        super().__init__(
            unique_id=unique_id_1,
            mass=mass_1,
            spin=spin_1,
            spin_angle=spin_angle_1,
            orb_a=orb_a_1,
            orb_inc=np.full(len(unique_id_1), 0),
            orb_ecc=np.full(len(unique_id_1), 0),
            orb_ang_mom=np.full(len(unique_id_1), 0),
            orb_arg_periapse=np.full(len(unique_id_1), 0),
            gen=gen_1,
            **kwargs
        )

    # Legacy reference to unique_id
    @property
    def unique_id_1(self):
        return self.unique_id

    @unique_id_1.setter
    def unique_id_1(self, new_value):
        self.unique_id = new_value

    # Legacy reference to mass_1
    @property
    def mass_1(self):
        return self.mass

    @mass_1.setter
    def mass_1(self, new_value):
        self.mass = new_value

    # Legacy reference to orb_a_1
    @property
    def orb_a_1(self):
        return self.orb_a

    @orb_a_1.setter
    def orb_a_1(self, new_value):
        self.orb_a = new_value

    # Legacy reference to spin_1
    @property
    def spin_1(self):
        return self.spin

    @spin_1.setter
    def spin_1(self, new_value):
        self.spin = new_value

    # Legacy reference to spin_angle_1
    @property
    def spin_angle_1(self):
        return self.spin_angle

    @spin_angle_1.setter
    def spin_angle_1(self, new_value):
        self.spin_angle = new_value

    # Legacy reference to gen_1
    @property
    def gen_1(self):
        return self.gen

    @gen_1.setter
    def gen_1(self, new_value):
        self.gen = new_value

    @property
    def mass_total(self):
        return self.mass_1 + self.mass_2

    @override
    def get_super_list(self) -> dict[str, npt.NDArray[Any]]:
        super_list = super().get_super_list()

        # Redundancy for legacy purposes
        super_list["unique_id_1"] = self.unique_id
        super_list["mass_1"] = self.mass
        super_list["orb_a_1"] = self.orb_a
        super_list["spin_1"] = self.spin
        super_list["spin_angle_1"] = self.spin_angle
        super_list["gen_1"] = self.gen

        super_list["unique_id_2"] = self.unique_id_2
        super_list["mass_2"] = self.mass_2
        super_list["orb_a_2"] = self.orb_a_2
        super_list["spin_2"] = self.spin_2
        super_list["spin_angle_2"] = self.spin_angle_2
        super_list["gen_2"] = self.gen_2
        super_list["bin_sep"] = self.bin_sep
        super_list["bin_ecc"] = self.bin_ecc
        super_list["bin_orb_a"] = self.bin_orb_a
        super_list["bin_orb_ang_mom"] = self.bin_orb_ang_mom
        super_list["bin_orb_inc"] = self.bin_orb_inc
        super_list["bin_orb_ecc"] = self.bin_orb_ecc
        super_list["time_to_merger_gw"] = self.time_to_merger_gw
        super_list["flag_merging"] = self.flag_merging
        super_list["time_merged"] = self.time_merged


        return super_list

    @override
    def add_objects(self, agnObjectArray: 'AGNBinaryBlackHoleArray'):
        super().add_objects(agnObjectArray)

        if not isinstance(agnObjectArray, AGNBinaryBlackHoleArray):
            raise Exception(f"Type Error: Unable to add {type(agnObjectArray)} objects to AGNBinaryBlackHoleArray.")

        self.unique_id_2 = np.concatenate((self.unique_id_2, agnObjectArray.unique_id_2))
        self.mass_2 = np.concatenate((self.mass_2, agnObjectArray.mass_2))
        self.orb_a_2 = np.concatenate((self.orb_a_2, agnObjectArray.orb_a_2))
        self.spin_2 = np.concatenate((self.spin_2, agnObjectArray.spin_2))
        self.spin_angle_2 = np.concatenate((self.spin_angle_2, agnObjectArray.spin_angle_2))
        self.bin_sep = np.concatenate((self.bin_sep, agnObjectArray.bin_sep))
        self.bin_orb_a = np.concatenate((self.bin_orb_a, agnObjectArray.bin_orb_a))
        self.time_to_merger_gw = np.concatenate((self.time_to_merger_gw, agnObjectArray.time_to_merger_gw))
        self.flag_merging = np.concatenate((self.flag_merging, agnObjectArray.flag_merging))
        self.time_merged = np.concatenate((self.time_merged, agnObjectArray.time_merged))
        self.bin_ecc = np.concatenate((self.bin_ecc, agnObjectArray.bin_ecc))
        self.gen_2 = np.concatenate((self.gen_2, agnObjectArray.gen_2))
        self.bin_orb_ang_mom = np.concatenate((self.bin_orb_ang_mom, agnObjectArray.bin_orb_ang_mom))
        self.bin_orb_inc = np.concatenate((self.bin_orb_inc, agnObjectArray.bin_orb_inc))
        self.bin_orb_ecc = np.concatenate((self.bin_orb_ecc, agnObjectArray.bin_orb_ecc))


class AGNMergedBlackHoleArray(AGNBinaryBlackHoleArray):
    def __init__(self,
                 mass_final: npt.NDArray[uuid] = np.array([]),
                 spin_final: npt.NDArray[uuid] = np.array([]),
                 spin_angle_final: npt.NDArray[uuid] = np.array([]),
                 chi_eff: npt.NDArray[uuid] = np.array([]),
                 chi_p: npt.NDArray[uuid] = np.array([]),
                 **kwargs):

        self.mass_final = mass_final
        self.spin_final = spin_final
        self.spin_angle_final = spin_angle_final
        self.chi_eff = chi_eff
        self.chi_p = chi_p

        super().__init__(**kwargs)

    @override
    def get_super_list(self) -> dict[str, npt.NDArray[Any]]:
        super_list = super().get_super_list()

        super_list["mass_final"] = self.mass_final
        super_list["spin_final"] = self.spin_final
        super_list["spin_angle_final"] = self.spin_angle_final
        super_list["chi_eff"] = self.chi_eff
        super_list["chi_p"] = self.chi_p

        return super_list

    @override
    def add_objects(self, agnObjectArray: 'AGNMergedBlackHoleArray'):
        super().add_objects(agnObjectArray)

        if not isinstance(agnObjectArray, AGNMergedBlackHoleArray):
            raise Exception(f"Type Error: Unable to add {type(agnObjectArray)} objects to AGNMergedBlackHoleArray.")

        self.mass_final = np.concatenate((self.mass_final, agnObjectArray.mass_final))
        self.spin_final = np.concatenate((self.spin_final, agnObjectArray.spin_final))
        self.spin_angle_final = np.concatenate((self.spin_angle_final, agnObjectArray.spin_angle_final))
        self.chi_eff = np.concatenate((self.chi_eff, agnObjectArray.chi_eff))
        self.chi_p = np.concatenate((self.chi_p, agnObjectArray.chi_p))


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