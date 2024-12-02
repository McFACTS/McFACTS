import numpy as np

from mcfacts.objects.simulation_actor import SimulationActor


class SimulationTimeline:
    """
    SimulationTimeline:
        A class representing the timeline of a simulation, including a sequence of simulation actors and their execution order.

        Attributes:
            name (str): The name of the simulation timeline.
            timesteps (int): The total number of timesteps in the simulation.
            timestep_length (float): The duration of each timestep.
            ordered_actors (dict[int, SimulationActor]): A dictionary mapping relative execution orders to SimulationActor instances.

        Methods:
            addSimulationActor(simulation_actor: SimulationActor, relative_order: int = 0) -> int:
                Adds a single SimulationActor to the timeline with the specified execution order.
            addSimulationActors(simulation_actors: list[SimulationActor]):
                Adds multiple SimulationActor instances to the timeline.
            getSimulationActor(relative_order: int) -> SimulationActor:
                Retrieves a SimulationActor based on its relative execution order.
            getOrderedActorList() -> list[SimulationActor]:
                Returns a list of SimulationActors sorted by their execution order.
            __str__() -> str:
                Returns a string representation of the timeline, including its name and ordered actors.
            __len__() -> int:
                Returns the number of simulation actors in the timeline.
    """
    def __init__(self, name: str, timesteps: int, timestep_length: float, ordered_actors: dict[int, SimulationActor] = None):
        """
        __init__(name: str, timesteps: int, timestep_length: float, ordered_actors: dict[int, SimulationActor] = None):
            Initializes the SimulationTimeline with a name, number of timesteps, and timestep duration.

            Parameters:
                name (str): The name of the simulation timeline.
                timesteps (int): The total number of timesteps in the simulation.
                timestep_length (float): The duration of each timestep in years.
                ordered_actors (dict[int, SimulationActor], optional): A dictionary of actors and their relative execution orders. Defaults to an empty dictionary.
        """
        self.name: str = name
        self.timesteps: int = timesteps
        self.timestep_length: float = timestep_length
        self.ordered_actors: dict[int, SimulationActor] = dict() if ordered_actors is None else ordered_actors


    def add_simulation_actor(self, simulation_actor: SimulationActor, relative_order: int = 0) -> int:
        """
        addSimulationActor(simulation_actor: SimulationActor, relative_order: int = 0) -> int:
            Adds a single SimulationActor to the timeline with the specified execution order.

            Parameters:
                simulation_actor (SimulationActor): The actor to add to the timeline.
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

        self.ordered_actors[relative_order] = simulation_actor

        return relative_order

    def add_simulation_actors(self, simulation_actors: list[SimulationActor]):
        """
        addSimulationActors(simulation_actors: list[SimulationActor]):
            Adds multiple SimulationActor instances to the timeline.

            Parameters:
                simulation_actors (list[SimulationActor]): A list of SimulationActor instances to add.
        """
        for simulation_actor in simulation_actors:
            self.add_simulation_actor(simulation_actor)


    def get_simulation_actor(self, relative_order: int) -> SimulationActor:
        """
        getSimulationActor(relative_order: int) -> SimulationActor:
            Retrieves a SimulationActor based on its relative execution order.

            Parameters:
                relative_order (int): The relative order of the desired SimulationActor.

            Returns:
                SimulationActor: The actor corresponding to the specified order.

            Raises:
                Exception: If no actor exists at the specified relative order.
        """
        if not (relative_order in self.ordered_actors):
            raise Exception(f"Could not find simulation actor at a relative order of {relative_order}.")

        return self.ordered_actors[relative_order]


    def get_ordered_actor_list(self) -> list[SimulationActor]:
        """
        getOrderedActorList() -> list[SimulationActor]:
            Returns a list of SimulationActors sorted by their execution order.

            Returns:
                list[SimulationActor]: A list of SimulationActors in execution order.
        """
        sorted_keys = sorted(self.ordered_actors.keys())

        sorted_actors: list[SimulationActor] = []

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