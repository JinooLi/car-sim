import numpy as np


class State:
    def __init__(self, x=0.0, y=0.0, theta=0.0, velocity=0.0):
        self.x = x
        self.y = y
        self.theta = theta
        self.velocity = velocity

    def __add__(self, other):
        return State(
            self.x + other.x,
            self.y + other.y,
            self.theta + other.theta,
            self.velocity + other.velocity,
        )

    def __mul__(self, other):
        return State(
            self.x * other.x,
            self.y * other.y,
            self.theta * other.theta,
            self.velocity * other.velocity,
        )

    def __mul__(self, scalar):
        return State(
            self.x * scalar,
            self.y * scalar,
            self.theta * scalar,
            self.velocity * scalar,
        )


class Input:
    def __init__(self, steer=0.0, acceleration=0.0):
        self.steer = steer
        self.acceleration = acceleration


class BicycleModel:
    def __init__(self, wheelbase=1.0):
        self.wheelbase = wheelbase

    def differential(self, state, input) -> State:
        """
        Calculate the differential state of the bicycle model.

        @param state: Current state of the bicycle (State object).

        @param input: Input to the bicycle model (Input object).

        @output: Returns the differential state (State object).
        """
        # Bicycle model differential equations
        dstate = State()
        dstate.x = state.velocity * np.cos(state.theta)
        dstate.y = state.velocity * np.sin(state.theta)
        dstate.theta = state.velocity * np.tan(input.steer) / self.wheelbase
        dstate.velocity = input.acceleration
        return dstate
