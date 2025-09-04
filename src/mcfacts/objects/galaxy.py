import copy
import os.path
import sys
from abc import ABC, abstractmethod

import numpy as np
from numpy.random import Generator
from tqdm.auto import tqdm

from mcfacts.inputs.settings_manager import SettingsManager, AGNDisk
from mcfacts.objects.snapshot import SnapshotHandler, TxtSnapshotHandler
from mcfacts.utilities.random_state import reset_random
from mcfacts.objects.agn_object_array import AGNObjectArray, FilingCabinet
from mcfacts.objects.timeline import SimulationTimeline


class GalaxyPopulator(ABC):
    def __init__(self, name: str, settings: SettingsManager = SettingsManager()):
        self.name: str = name
        self.settings_manager: SettingsManager = settings

    @abstractmethod
    def populate(self, agn_disk: AGNDisk, random_generator: Generator) -> AGNObjectArray:
        return NotImplemented


class Galaxy:
    """
    Galaxy:
        A class representing a galaxy in a simulation, with capabilities to populate AGN objects and run simulations.

        Attributes:
            seed (int): Random seed for generating reproducible results.
            runs_folder (string): Path to where the containing folder for the output folder should be.
            settings (SettingsManager): Configuration settings for the galaxy.
            galaxy_id (string): Id or name used to create the output folder for the simulation.
            random_generator (np.random.Generator): Random state instance for controlling randomness.
            filing_cabinet (dict[str, AGNObjectArray]): A dictionary storing AGNObjectArray objects organized by populator name.
            timeline_history (list[SimulationTimeline]): A history of simulation timelines executed by the galaxy.
            populated (bool): A flag indicating whether the galaxy has been populated with objects.

        Methods:
            populate(populators: list[GalaxyPopulator], strict_fill: bool = True, join_populations: bool = False):
                Populates the galaxy using a list of GalaxyPopulator instances.
            simulate(simulationTimeline: SimulationTimeline):
                Runs a simulation using the specified SimulationTimeline, updating the state of the galaxy.
    """

    def __init__(self, seed: int, runs_folder: str, galaxy_id: str, settings: SettingsManager = SettingsManager(), snapshot_handler: SnapshotHandler = None):
        """
        __init__(settings_manger: SettingsManager, seed: int):
            Initializes the Galaxy instance with configuration settings and a random seed.

            Parameters:
                seed (int): Random seed for reproducible randomness.
                runs_folder (string): Path to where the containing folder for the output folder should be.
                settings (SettingsManager, optional): Configuration settings for the galaxy.
                galaxy_id (string): Id or name used to create the output folder for the simulation.
        """
        self.seed: int = seed
        self.runs_folder: str = runs_folder
        self.settings: SettingsManager = settings
        self.galaxy_id: str = galaxy_id

        # Setup random
        self.random_generator = np.random.default_rng(seed)

        # Set seed in mcfacts_random_state so non legacy methods are consistent.
        # TODO: Make all methods using global random from mcfacts_random_state use galaxy randomGenerator through pass by reference.
        reset_random(int(str(seed)[len(str(seed)) - 10:]))

        self.filing_cabinet: FilingCabinet = FilingCabinet()
        self.timeline_history: list[SimulationTimeline] = list()
        self.populated: bool = False

        self.snapshot_handler = snapshot_handler

        if snapshot_handler is None:
            self.snapshot_handler = TxtSnapshotHandler(self.settings)

        # Set the recursion limit higher so python doesn't scream at us. The timeline-actor framework does not do any recursion,
        # but when an actor performs, something can end up executing several "layers" away from the initial call.
        sys.setrecursionlimit(10000)

    def save_state(self, timestep: int = None) -> None:
        save_folder = os.path.join(self.runs_folder, f"galaxy_{self.galaxy_id}")
        file_name = f"galaxy_state_{len(self.timeline_history)}"

        if timestep is not None:
            history = len(self.timeline_history)

            save_folder = os.path.join(save_folder, f"galaxy_state_{history - 1}_to_{history}")
            file_name = f"galaxy_state_{history - 1}_to_{history}_T{timestep}"

        self.log(f"Saving state of galaxy to {save_folder} as {file_name}")

        self.snapshot_handler.save_cabinet(save_folder, file_name, self.filing_cabinet)

    def populate(self, populators: list[GalaxyPopulator], agn_disk: AGNDisk, strict_fill: bool = True, join_populations: bool = False) -> None:
        """
        populate(populators: list[GalaxyPopulator], strict_fill: bool = True, join_populations: bool = False):
            Populates the galaxy with AGN objects using the provided GalaxyPopulator instances.

            Parameters:
                populators (list[GalaxyPopulator]): A list of GalaxyPopulator instances to generate AGN objects.
                agn_disk (AGNDisk): An object containing the generated properties of the AGN disk.
                strict_fill (bool, optional): If True, ensures all populators create objects. Raises an exception if no objects are created. Defaults to True.
                join_populations (bool, optional): If True, combines existing populations with new ones for the same populator name. Defaults to False.

            Raises:
                Exception: If strict_fill is True and no populators are provided, or if a populator fails to create objects.
                Exception: If a populator with the same name already exists in agn_objects (unless join_populations is True).
        """
        if strict_fill and len(populators) == 0:
            raise Exception("No populators have been passed into the method.")

        self.log("Beginning galaxy population.")

        # Loop over the populators
        for populator in populators:
            galaxy_object_array: AGNObjectArray = populator.populate(agn_disk, self.random_generator)

            if join_populations:
                self.filing_cabinet.create_or_append_array(populator.name, galaxy_object_array)
            else:
                if populator.name in self.filing_cabinet:
                    raise Exception(f"Galaxy populator with name {populator.name} already exist.")

                self.filing_cabinet.set_array(populator.name, galaxy_object_array)

                self.log("Beginning galaxy population.")

            # In strict mode, check to make sure that we actually created some objects, otherwise throw an exception.
            if strict_fill and len(galaxy_object_array) == 0:
                raise Exception(f"Galaxy populator with name {populator.name} failed to create any object populations.")

        self.log("Checking filing cabinet for duplicate entries.")
        self.filing_cabinet.consistency_check()

        self.populated = True

        if self.settings.save_state:
            self.save_state()

    def run(self, simulation_timeline: SimulationTimeline, agn_disk: AGNDisk) -> None:
        """
        run(simulation_timeline: SimulationTimeline):
            Executes a list of actions based on the given SimulationTimeline, updating the state of the galaxy and saving the timeline.

            Parameters:
                simulation_timeline (SimulationTimeline): The timeline of events and actors for the simulation.
                agn_disk (AGNDisk): An object containing the generated properties of the AGN disk.

            Side Effects:
                - Adds a copy of the simulation timeline to the timeline_history.
                - Iteratively calls perform() on actors in the simulation at each timestep.
        """
        if not self.populated:
            raise Exception("Unable to progress through a timeline as the galaxy has not been populated yet.")

        # Create a copy of the timeline with all settings and save it to the history
        active_timeline = copy.deepcopy(simulation_timeline)
        self.timeline_history.append(active_timeline)

        for timestep in tqdm(range(active_timeline.timesteps), desc=f"(ID:{self.galaxy_id}) Running {active_timeline.name}", position=0, leave=True, disable=False if self.settings.show_timeline_progress else True):
            timestep_length = active_timeline.timestep_length
            time_passed = timestep * timestep_length

            self.log("---------------------")
            self.log(f"Timestep: {timestep}, Time passed: {time_passed}", False)

            for actor in active_timeline.get_ordered_actor_list():

                if actor.settings is None:
                    actor.settings = self.settings

                self.log(f"<T:{timestep}> Running {actor.name}, Using Galaxy Settings: {actor.settings.settings_overrides == self.settings.settings_overrides}")

                actor.set_log_func(self.nocheck_log)
                actor.perform(timestep, timestep_length, time_passed, self.filing_cabinet, agn_disk, self.random_generator)

            if self.settings.save_each_timestep:
                self.save_state(timestep)

        # Only run a filing cabinet consistency check once per run, since we check every entry against every other entry in the cabinet O(n^2).
        self.log(f"Checking filing cabinet for duplicate entries.")
        self.filing_cabinet.consistency_check()

        if self.settings.save_state:
            self.save_state()

    def nocheck_log(self, msg: str, new_line: bool = True) -> None:
        print(f"{(os.linesep if new_line else '')}(ID:{self.galaxy_id}) {msg}")

    def log(self, msg: str, new_line: bool = False) -> None:
        if not self.settings.verbose:
            return

        self.nocheck_log(msg, new_line)
