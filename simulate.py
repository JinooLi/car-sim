from src.interface import Controller, Input, Model, SimulateResult, State, Visualizer


class Simulator:
    def __init__(
        self,
        model: Model,  # Model to simulate
        controller: Controller,  # Controller for managing inputs to the model
        simulation_time: float = 10.0,  # Total simulation time in seconds
        time_step: float = 0.01,  # Time step for simulation updates
    ):
        self.model = model
        self.controller = controller
        self.simulation_time = simulation_time
        self.time_step = time_step
        self.time_steps = int(simulation_time / time_step)
        self.control_time_step = controller.controller_time_step

    def simulate(self, initial_state: State) -> SimulateResult:
        state = initial_state
        t = 0.0
        control_time = 0.0
        result = SimulateResult(self.simulation_time, self.time_step)
        print(f"Starting simulation")
        result.append_state(state)
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
