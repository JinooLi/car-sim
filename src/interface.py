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


class Simulator(ABC):
    def __init__(
        self,
        model: Model,  # Model to simulate
        controller: Controller,  # Controller for managing inputs to the model
        initial_state: State,  # Initial state of the model
        simulation_time: float = 10.0,  # Total simulation time in seconds
        time_step: float = 0.01,  # Time step for simulation updates
    ):
        self.model = model
        self.controller = controller
        self.initial_state = initial_state
        self.simulation_time = simulation_time
        self.time_step = time_step
        self.time_steps = int(simulation_time / time_step)
        self.control_time_step = controller.controller_time_step

    def simulate(self) -> SimulateResult:
        state = self.initial_state
        t = 0.0
        control_time = 0.0
        result = SimulateResult(self.simulation_time, self.time_step)
        print(f"Starting simulation")
        for _ in range(self.time_steps):
            # Update control signal at specified intervals(self.control_time_step)
            if t >= control_time:
                input_signal = self.controller.control(state)
                control_time += self.control_time_step
            state = self.__RK4_step(state, input_signal)
            result.append_state(state)
            t += self.time_step
        print(f"Simulation completed.")
        return result

    def __RK4_step(self, state: State, input_signal: Input):
        k1 = self.model.differential(state, input_signal)
        k2 = self.model.differential(state + k1 * (self.time_step / 2), input_signal)
        k3 = self.model.differential(state + k2 * (self.time_step / 2), input_signal)
        k4 = self.model.differential(state + k3 * self.time_step, input_signal)
        return state + (k1 + 2 * k2 + 2 * k3 + k4) * (self.time_step / 6)


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
