from abc import ABC, abstractmethod


class State(ABC):
    @abstractmethod
    def __init__(self):
        # Initialize state variables
        pass

    @abstractmethod
    def __add__(self, other):
        # Add two states together
        pass

    @abstractmethod
    def __mul__(self, other):
        # Multiply two states together
        pass

    @abstractmethod
    def __truediv__(self, scalar: float | int):
        # Divide state by a scalar
        pass


class Input(ABC):
    @abstractmethod
    def __init__(self):
        # Initialize input variables
        pass


class Model(ABC):
    @abstractmethod
    def __init__(self):
        # Initialize model parameters
        pass

    @abstractmethod
    def differential(self, state: State, input: Input) -> State:
        """
        Calculate the differential state of the model.

        @param state: Current state of the model (State object).
        @param input: Input to the model (Input object).
        @return: Returns the differential state (State object).
        """
        pass


class Controller(ABC):
    @abstractmethod
    def __init__(self, model: Model, controller_time_step: float = 0.01):
        # Initialize controller parameters
        self.model = model
        self.controller_time_step = controller_time_step

    @abstractmethod
    def control(self, state: State) -> Input:
        pass


class SimulateResult(ABC):
    def __init__(self, simulation_time: float, time_step: float):
        # Initialize simulation result parameters
        self.simulation_time = simulation_time
        self.time_step = time_step
        self.states = []

    def append_state(self, state: State):
        """
        Append a state to the simulation results.
        """
        self.states.append(state)

    def get_results(self) -> tuple[list[State], float, float]:
        """Get the results of the simulation, including the states, simulation time, and time step.

        Returns:
            - tuple
            [list[State], float, float]: A tuple containing the list of states, simulation time, and time step.
        """
        return self.states, self.simulation_time, self.time_step


class Visualizer(ABC):
    @abstractmethod
    def __init__(self, fps: int = 30):
        # Initialize visualizer parameters
        self.fps = fps

    @abstractmethod
    def visualize(
        self,
        data: SimulateResult,
    ):
        """
        Visualize the current state of the model.

        @param data: Simulation results (SimulateResult object).
        """
        pass
