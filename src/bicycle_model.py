import numpy as np
from .interface import State, Input, Model, Controller, Visualizer, SimulateResult


class BicycleState(State):
    def __init__(self, x=0.0, y=0.0, theta=0.0, velocity=0.0):
        super().__init__()
        self.x = x
        self.y = y
        self.theta = theta
        self.velocity = velocity

    def __add__(self, other):
        return BicycleState(
            self.x + other.x,
            self.y + other.y,
            self.theta + other.theta,
            self.velocity + other.velocity,
        )

    __radd__ = __add__

    def __mul__(self, other):
        if isinstance(other, (float, int)):
            return BicycleState(
                self.x * other,
                self.y * other,
                self.theta * other,
                self.velocity * other,
            )
        elif isinstance(other, BicycleState):
            # Element-wise multiplication for two states
            return BicycleState(
                self.x * other.x,
                self.y * other.y,
                self.theta * other.theta,
                self.velocity * other.velocity,
            )
        else:
            raise TypeError("Unsupported type for multiplication with BicycleState")

    __rmul__ = __mul__

    def __truediv__(self, other) -> "BicycleState":
        if isinstance(other, (float, int)):
            return BicycleState(
                self.x / other,
                self.y / other,
                self.theta / other,
                self.velocity / other,
            )
        elif isinstance(other, BicycleState):
            # Element-wise division for two states
            return BicycleState(
                self.x / other.x,
                self.y / other.y,
                self.theta / other.theta,
                self.velocity / other.velocity,
            )
        else:
            raise TypeError("Unsupported type for division with BicycleState")


class BicycleInput(Input):
    def __init__(self, steer=0.0, acceleration=0.0):
        super().__init__()
        self.steer = steer
        self.acceleration = acceleration


class BicycleModel(Model):
    def __init__(self, wheelbase=1.0):
        super().__init__()
        self.wheelbase = wheelbase

    def differential(self, state: BicycleState, input: BicycleInput) -> State:
        """
        Calculate the differential state of the bicycle model.

        @param state: Current state of the bicycle (State object).

        @param input: Input to the bicycle model (Input object).

        @output: Returns the differential state (State object).
        """
        # Bicycle model differential equations
        dstate = BicycleState()
        dstate.x = state.velocity * np.cos(state.theta)
        dstate.y = state.velocity * np.sin(state.theta)
        dstate.theta = state.velocity * np.tan(input.steer) / self.wheelbase
        dstate.velocity = input.acceleration
        return dstate


class BicycleController(Controller):
    def __init__(self, target_velocity=1.0, control_time_step=0.01):
        super().__init__(control_time_step)
        self.target_velocity = target_velocity

    def control(self, state: BicycleState) -> BicycleInput:
        """
        Generate control input based on the current state.

        @param state: Current state of the bicycle (State object).
        @return: Returns the control input (Input object).
        """
        steer = 0.1  # Placeholder for steering logic
        acceleration = self.target_velocity - state.velocity
        return BicycleInput(steer=steer, acceleration=acceleration)


class BicycleVisualizer(Visualizer):
    def __init__(self, fps: int = 30):
        super().__init__(fps)

    def visualize(self, data: SimulateResult):
        """
        Visualize the simulation results.

        @param data: Simulation results (SimulateResult object).
        """
        print("Visualizing simulation results:")
