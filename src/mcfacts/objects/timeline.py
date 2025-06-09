from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator

from mcfacts.inputs.settings_manager import SettingsManager, AGNDisk
from mcfacts.objects.galaxy import FilingCabinet


class TimelineActor(ABC):
    """
    A base class representing a generic timeline actor in the simulation framework.

    Timeline actors define behaviors or actions to be performed during the simulation,
    and each actor is identified by a name and utilizes specific settings.

    Attributes:
        name (str): The name of the simulation actor.
        settings (SettingsManager): A manager for accessing configuration settings. Defaults to the base `SettingsManager`.

    Methods:
        perform(timestep: int, filing_cabinet: FilingCabinet, randomState: RandomState):
            Abstract method to be implemented by subclasses. Defines the behavior of the simulation actor for a given timestep.

        __str__():
            Returns a string representation of the simulation actor, including its name and type.
    """

    def __init__(self, name: str, settings: SettingsManager = None):
        """
        Initializes a timelineActor instance.

        Args:
            name (str): The name of the simulation actor.
            settings (SettingsManager, optional): The settings manager to retrieve configurations for the actor.
                Defaults to an empty `SettingsManager`.
        """
        self.name: str = name
        self.settings: SettingsManager = settings
        self.parent_log_func = None

    @abstractmethod
    def perform(self, timestep: int, timestep_length: float, time_passed: float, filing_cabinet: FilingCabinet, agn_disk: AGNDisk, random_generator: Generator):
        """
        Abstract method to define the behavior of the simulation actor during a timestep.

        Subclasses must implement this method to specify actions to be taken in the simulation.

        Args:
            timestep (int): The current simulation timestep.
            timestep_length (float): The duration of a timestep in years.
            time_passed (float): Time passed in years.
            filing_cabinet (FilingCabinet): A container for accessing and modifying simulation data arrays.
            agn_disk (AGNDisk): An object containing the generated properties of the AGN disk.
            random_generator (Generator): A random number generator for stochastic behaviors.

        Returns:
            NotImplemented: Must be overridden by subclasses.
        """
        return NotImplemented

    def set_log_func(self, log_func: callable):
        self.parent_log_func = log_func

    def log(self, msg: str, new_line: bool = False):
        if not self.settings.verbose:
            return

        if self.parent_log_func is None:
            print(f"{"\n" if new_line else ""}(ID:??) {msg}")
        else:
            self.parent_log_func(msg, new_line)

    def __str__(self):
        """
        Returns a string representation of the simulation actor.

        The string includes the actor's name and its class type.

        Returns:
            str: A formatted string representation of the simulation actor.
        """
        return f"{self.name} ({type(self)})"


class SimulationTimeline:
    """
    SimulationTimeline:
        A class representing the timeline of a simulation, including a sequence of simulation actors and their execution order.

        Attributes:
            name (str): The name of the simulation timeline.
            timesteps (int): The total number of timesteps in the simulation.
            timestep_length (float): The duration of each timestep.
            ordered_actors (dict[int, TimelineActor]): A dictionary mapping relative execution orders to timelineActor instances.

        Methods:
            addtimelineActor(timeline_actor: timelineActor, relative_order: int = 0) -> int:
                Adds a single timelineActor to the timeline with the specified execution order.
            addtimelineActors(timeline_actors: list[timelineActor]):
                Adds multiple timelineActor instances to the timeline.
            gettimelineActor(relative_order: int) -> timelineActor:
                Retrieves a timelineActor based on its relative execution order.
            getOrderedActorList() -> list[timelineActor]:
                Returns a list of timelineActors sorted by their execution order.
            __str__() -> str:
                Returns a string representation of the timeline, including its name and ordered actors.
            __len__() -> int:
                Returns the number of simulation actors in the timeline.
    """
    def __init__(self, name: str, timesteps: int, timestep_length: float, ordered_actors: dict[int, TimelineActor] = None):
        """
        __init__(name: str, timesteps: int, timestep_length: float, ordered_actors: dict[int, timelineActor] = None):
            Initializes the SimulationTimeline with a name, number of timesteps, and timestep duration.

            Parameters:
                name (str): The name of the simulation timeline.
                timesteps (int): The total number of timesteps in the simulation.
                timestep_length (float): The duration of each timestep in years.
                ordered_actors (dict[int, timelineActor], optional): A dictionary of actors and their relative execution orders. Defaults to an empty dictionary.
        """
        self.name: str = name
        self.timesteps: int = timesteps
        self.timestep_length: float = timestep_length
        self.ordered_actors: dict[int, TimelineActor] = dict() if ordered_actors is None else ordered_actors


    def add_timeline_actor(self, timeline_actor: TimelineActor, relative_order: int = 0) -> int:
        """
        addtimelineActor(timeline_actor: timelineActor, relative_order: int = 0) -> int:
            Adds a single timelineActor to the timeline with the specified execution order.

            Parameters:
                timeline_actor (TimelineActor): The actor to add to the timeline.
                relative_order (int, optional): The relative order for the actor's execution. Defaults to 0. If set to 0 and actors already exist, a new order is assigned.

            Returns:
                int: The relative order assigned to the added actor.

            Raises:
                Exception: If an actor already exists at the specified relative order.
        """
        if len(self.ordered_actors) > 0 and relative_order == 0:
            relative_order = np.max(list(self.ordered_actors.keys())) + 1000

        if relative_order in self.ordered_actors:
            raise Exception(f"Could not add simulation actor as one already exists with the relative order of {relative_order}.")

        self.ordered_actors[relative_order] = timeline_actor

        return relative_order

    def add_timeline_actors(self, timeline_actors: list[TimelineActor]):
        """
        addtimelineActors(timeline_actors: list[timelineActor]):
            Adds multiple timelineActor instances to the timeline.

            Parameters:
                timeline_actors (list[TimelineActor]): A list of timelineActor instances to add.
        """
        for timeline_actor in timeline_actors:
            self.add_timeline_actor(timeline_actor)


    def get_timeline_actor(self, relative_order: int) -> TimelineActor:
        """
        gettimelineActor(relative_order: int) -> timelineActor:
            Retrieves a timelineActor based on its relative execution order.

            Parameters:
                relative_order (int): The relative order of the desired timelineActor.

            Returns:
                TimelineActor: The actor corresponding to the specified order.

            Raises:
                Exception: If no actor exists at the specified relative order.
        """
        if not (relative_order in self.ordered_actors):
            raise Exception(f"Could not find simulation actor at a relative order of {relative_order}.")

        return self.ordered_actors[relative_order]


    def get_ordered_actor_list(self) -> list[TimelineActor]:
        """
        getOrderedActorList() -> list[TimelineActor]:
            Returns a list of timelineActors sorted by their execution order.

            Returns:
                list[TimelineActor]: A list of timelineActors in execution order.
        """
        sorted_keys = sorted(self.ordered_actors.keys())

        sorted_actors: list[TimelineActor] = []

        for relative_order in sorted_keys:
            sorted_actors.append(self.ordered_actors[relative_order])

        return sorted_actors


    def __str__(self):
        """
        __str__() -> str:
            Returns a string representation of the simulation timeline, including its name and the ordered actors.

            Returns:
                str: A formatted string showing the timeline name and the actors with their execution orders.
        """
        sorted_keys = sorted(self.ordered_actors.keys())

        string_builder = f"Simulation Timeline: {self.name}\n"

        for relative_order in sorted_keys:
            string_builder = string_builder + f"Order: {relative_order}, Actor: {self.ordered_actors[relative_order]}\n"

        return string_builder

    def __len__(self):
        """
        __len__() -> int:
            Returns the number of simulation actors in the timeline.

            Returns:
                int: The number of actors in the timeline.
        """
        return len(self.ordered_actors)