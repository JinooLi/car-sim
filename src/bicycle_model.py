import numpy as np
from .interface import State, Input, Model, Controller, Visualizer, SimulateResult
import matplotlib.pyplot as plt
from matplotlib import animation


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
    def __init__(
        self,
        model: BicycleModel,
        target_position: tuple[float, float],
        control_time_step=0.1,
    ):
        """Initialize the bicycle controller.

        Args:
            model (BicycleModel): The bicycle model.
            target_position (tuple[float, float]): The target position (x, y) for the bicycle.
            control_time_step (float, optional): The time step for control updates. Defaults to 0.1.
        """
        super().__init__(model, control_time_step)
        self.target_position = target_position

    def control(self, state: BicycleState) -> BicycleInput:
        """Generate control input based on the current state.

        Args:
            state (BicycleState): The current state of the bicycle.

        Returns:
            BicycleInput: _description_
        """

        target_velocity, steer = self.__ref_controller(
            state, gamma=1.0, beta=2.9, h=2.0
        )

        acceleration = 10 * (target_velocity - state.velocity)
        return BicycleInput(steer=steer, acceleration=acceleration)

    def __ref_controller(
        self, state: BicycleState, gamma: float, beta: float, h: float
    ) -> tuple[float, float]:
        """Calculate the reference control inputs for the bicycle model.

        h > 0, 2 < beta < h + 1

        Args:
            state (BicycleState): Current state of the bicycle.
            gamma (float): Proportional gain for position control.
            beta (float): Proportional gain for steering control.
            h (float): Feedforward gain for steering control.

        Returns:
            tuple[float, float]: A tuple containing the velocity and steering angle.
        """
        tp = self.target_position
        x = state.x
        y = state.y
        theta = state.theta

        pos_error = np.sqrt((tp[0] - x) ** 2 + (tp[1] - y) ** 2)
        goal_angle = np.arctan2(tp[1] - y, tp[0] - x)
        steer_error = goal_angle - theta

        velocity = gamma * pos_error
        if velocity > 5:
            velocity = 5  # Limit maximum velocity

        c = (
            np.sin(steer_error)
            + h * goal_angle * np.sin(steer_error) / (steer_error)
            + beta * steer_error
        ) / pos_error

        steer_angle = np.arctan(c * self.model.wheelbase)

        if pos_error < 0.1:  # Stop if close to target
            steer_angle = 0.0
            velocity = 0.0

        return velocity, steer_angle


class BicycleVisualizer(Visualizer):
    def __init__(self, model: BicycleModel, fps: int = 30):
        """ """
        super().__init__(fps)
        self.model = model

    def visualize(self, data: SimulateResult):
        """
        Visualize the simulation results.

        @param data: Simulation results (SimulateResult object).
        """

        states, simulation_time, time_step = data.get_results()

        print(f"Visualizing simulation results with {len(states)} states.")

        state_x = [state.x for state in states]
        state_y = [state.y for state in states]

        car_length = self.model.wheelbase
        car_width = 0.5 * car_length

        max_x = (max(state_x) if state_x else 0) + car_length
        max_y = (max(state_y) if state_y else 0) + car_length
        min_x = (min(state_x) if state_x else 0) - car_length
        min_y = (min(state_y) if state_y else 0) - car_length

        fig, ax = plt.subplots()
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        x_len = max_x - min_x
        y_len = max_y - min_y
        ratio = y_len / x_len  # x_len cant be zero.
        radius_inch = 12
        x_size = np.sqrt(radius_inch**2 / (1 + ratio**2))
        y_size = ratio * x_size
        fig.set_size_inches(x_size, y_size)

        car = plt.Rectangle(
            (0, 0), car_length, car_width, angle=0, color="blue", alpha=0.5
        )
        ax.add_patch(car)

        def init():
            car.set_xy((0, -car_width / 2))
            return (car,)

        x_fps_history = []
        y_fps_history = []
        theta_fps_history = []
        t = 0
        frame_interval = 1 / self.fps
        x_fps_history.append(states[0].x)
        y_fps_history.append(states[0].y)
        theta_fps_history.append(states[0].theta)
        for i in range(len(states)):
            t += time_step
            if t >= frame_interval:
                x_fps_history.append(states[i].x)
                y_fps_history.append(states[i].y)
                theta_fps_history.append(states[i].theta)
                t -= frame_interval

        def animate(i):
            if i < len(x_fps_history):
                x = x_fps_history[i]
                y = y_fps_history[i]
                theta = theta_fps_history[i]
            else:
                x = x_fps_history[-1]
                y = y_fps_history[-1]
                theta = theta_fps_history[-1]
            car.set_xy((x, y - car_width / 2))
            car.angle = np.degrees(theta)

            return (car,)

        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=len(x_fps_history),
            init_func=init,
            interval=1000 * frame_interval,
            blit=True,
        )

        print(len(x_fps_history))

        print("Saving animation to bicycle_simulation.mp4")
        ani.save("bicycle_simulation.mp4", writer="ffmpeg", fps=self.fps)
