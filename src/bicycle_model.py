import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from cvxopt import matrix, solvers
from matplotlib import animation

from .interface import (
    Controller,
    Input,
    Model,
    SimulateResult,
    State,
    Simulator,
    Visualizer,
)


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
        """Calculate the differential state of the bicycle model.

        Args:
            state (BicycleState): Current state of the bicycle.
            input (BicycleInput): Input to the bicycle model.

        Returns:
            State: Differential state of the bicycle.
        """
        # Bicycle model differential equations
        dstate = BicycleState()
        dstate.x = state.velocity * np.cos(state.theta)
        dstate.y = state.velocity * np.sin(state.theta)
        dstate.theta = state.velocity * np.tan(input.steer) / self.wheelbase
        dstate.velocity = input.acceleration
        return dstate


class Obstacle:
    def __init__(self, position: tuple[float, float], radius: float):
        """Initialize an obstacle.

        Args:
            position (tuple[float, float]): The (x, y) position of the obstacle.
            radius (float): The radius of the obstacle.
        """
        self.position = position
        self.radius = radius


class BicycleController(Controller):
    def __init__(
        self,
        model: BicycleModel,
        target_position: tuple[float, float],
        target_angle: float,
        obstacle: Obstacle,
        controller_time_step: float = 0.1,
        filter: bool = True,
        steer_limit: bool = True,
        k1: float = 10.0,
        k2: float = 10.0,
        k3: float = 5.0,
    ):
        """Initialize the bicycle controller.

        Args:
            model (BicycleModel): The bicycle model.
            target_position (tuple[float, float]): The target position (x, y) for the bicycle.
            target_angle (float): The target angle for the bicycle.
            obstacle (Obstacle, optional): The obstacle to avoid.
            controller_time_step (float, optional): The time step for control updates. Defaults to 0.1.
            filter (bool, optional): Whether to apply a safety filter. Defaults to True.
            steer_limit (bool, optional): Whether to limit the steering angle. Defaults to True.
            k1 (float, optional): First Gain for the barrier function. Defaults to 10.0.
            k2 (float, optional): Second Gain for the barrier function. Defaults to 10.0.
            k3 (float, optional): Third Gain for the barrier function. Defaults to 5.0.
        """
        super().__init__(model, controller_time_step)
        self.__init_safety_filter(obstacle, k1, k2, k3)
        self.target_position = target_position
        self.target_angle = target_angle
        self.filter = filter
        self.steer_limit = steer_limit
        self.barrier_data = []

    def control(self, state: BicycleState) -> BicycleInput:
        """Generate control input based on the current state.

        Args:
            state (BicycleState): The current state of the bicycle.

        Returns:
            BicycleInput: _description_
        """

        # Rotate the state and target position to align with the target angle
        rotation_matrix = np.array(
            [
                [np.cos(-self.target_angle), -np.sin(-self.target_angle)],
                [np.sin(-self.target_angle), np.cos(-self.target_angle)],
            ]
        )

        position_vector = np.array([state.x, state.y])
        target_position_vector = np.array(self.target_position)

        position_transformed = rotation_matrix @ position_vector
        target_transformed = tuple(rotation_matrix @ target_position_vector)

        state_transformed = BicycleState(
            x=position_transformed[0],
            y=position_transformed[1],
            theta=state.theta - self.target_angle,
            velocity=state.velocity,
        )

        target_velocity, steer = self.__ref_controller(
            state_transformed,
            max_speed=5.0,
            target_position=target_transformed,
            gamma=1.0,
            beta=2.9,
            h=2.0,
        )

        # Apply safety filter when the option is enabled
        if self.filter:
            target_velocity, steer = self.__safety_filter(
                state, target_velocity, steer, state.velocity, self.__prev_steer_angle
            )

        target_velocity = np.clip(target_velocity, 0.0, 5.0)  # Limit max speed
        if self.steer_limit:  # Limit steering angle
            steer = np.clip(steer, -np.pi / 6, np.pi / 6)

        pos_error = np.hypot(
            self.target_position[0] - state.x,
            self.target_position[1] - state.y,
        )
        angle_error = np.atan2(
            np.sin(self.target_angle - state.theta),
            np.cos(self.target_angle - state.theta),
        )
        if np.hypot(pos_error, angle_error) < 0.15:  # Stop if close to target
            steer = 0.0
            target_velocity = 0.0

        # save previous angle for safety filter
        self.__prev_steer_angle = steer

        acceleration = 10 * (target_velocity - state.velocity)  # simple P controller
        return BicycleInput(steer=steer, acceleration=acceleration)

    def __ref_controller(
        self,
        state: BicycleState,
        max_speed: float,
        target_position: tuple[float, float] = (0.0, 0.0),
        gamma: float = 1.0,
        beta: float = 2.9,
        h: float = 2.0,
    ) -> tuple[float, float]:
        """Calculate the reference control inputs for the bicycle model.

        h > 0, 2 < beta < h + 1

        Args:
            state (BicycleState): Current state of the bicycle.
            max_speed (float): Maximum speed of the bicycle.
            target_position (tuple[float, float]): Target position (x, y) for the bicycle.
            gamma (float): Proportional gain for position control.
            beta (float): Proportional gain for steering control.
            h (float): Feedforward gain for steering control.

        Returns:
            tuple[float, float]: A tuple containing the velocity and steering angle.
        """
        tp = target_position
        x = state.x
        y = state.y
        theta = state.theta

        pos_error = np.sqrt((tp[0] - x) ** 2 + (tp[1] - y) ** 2)
        goal_angle = np.arctan2(tp[1] - y, tp[0] - x)
        steer_error = goal_angle - theta
        steer_error = np.arctan2(
            np.sin(steer_error), np.cos(steer_error)
        )  # Normalize steer error

        velocity = gamma * pos_error

        if velocity > max_speed:
            velocity = max_speed  # Limit maximum velocity

        c = (
            np.sin(steer_error)
            + h * goal_angle * np.sin(steer_error) / steer_error
            + beta * steer_error
        ) / pos_error

        steer_angle = np.arctan(c * self.model.wheelbase)

        return velocity, steer_angle

    def __init_safety_filter(self, obstacle: Obstacle, k1: float, k2: float, k3: float):
        """Initialize the safety filter parameters.

        Args:
            obstacle (Obstacle): The obstacle to avoid.
            k1 (float): The parameter for the first barrier function.
            k2 (float): The parameter for the second barrier function.
            k3 (float): The parameter for the third barrier function.
        """
        print("Initializing safety filter...")
        t = sp.symbols("t")
        a = sp.symbols("a", real=True)
        omega = sp.symbols("omega", real=True)
        x = sp.Function("x", real=True)(t)
        y = sp.Function("y", real=True)(t)
        v = sp.Function("v", real=True)(t)  # velocity
        phi = sp.Function("phi", real=True)(t)  # heading angle
        psi = sp.Function("psi", real=True)(t)  # steering angle
        l = self.model.wheelbase

        dotx = v * sp.cos(phi)
        doty = v * sp.sin(phi)
        dotphi = v * sp.tan(psi) / l
        dotv = a
        dotpsi = omega

        input_symbols = (x, y, v, phi, psi, a, omega)

        # Define the safety condition
        h = (
            (x - obstacle.position[0]) ** 2
            + (y - obstacle.position[1]) ** 2
            - obstacle.radius**2
        )
        self.h = sp.lambdify(input_symbols, h, modules="numpy")

        # Define differential equations
        substitutions = {
            x.diff(t): dotx,
            y.diff(t): doty,
            phi.diff(t): dotphi,
            v.diff(t): dotv,
            psi.diff(t): dotpsi,
        }

        h_1 = h.diff(t) + k1 * h
        h_1 = h_1.subs(substitutions)
        self.h1 = sp.lambdify(input_symbols, h_1, modules="numpy")
        h_2 = h_1.diff(t) + k2 * h_1
        h_2 = h_2.subs(substitutions).simplify()
        self.h2 = sp.lambdify(input_symbols, h_2, modules="numpy")
        h_3 = h_2.diff(t) + k3 * h_2
        h_3 = h_3.subs(substitutions).simplify()
        self.h3 = sp.lambdify(input_symbols, h_3, modules="numpy")

        # t에 대한 function들을 모두 변수로 치환
        substitutions = {
            x: sp.symbols("x", real=True),
            y: sp.symbols("y", real=True),
            v: sp.symbols("v", real=True),
            phi: sp.symbols("phi", real=True),
            psi: sp.symbols("psi", real=True),
        }

        x = sp.symbols("x", real=True)
        y = sp.symbols("y", real=True)
        v = sp.symbols("v", real=True)
        phi = sp.symbols("phi", real=True)
        psi = sp.symbols("psi", real=True)

        h_3 = h_3.subs(substitutions)

        constant_term = h_3.subs({a: 0, omega: 0})
        coeff_a = (h_3 - h_3.subs({a: 0})).simplify().subs({a: 1})
        coeff_omega = (h_3 - h_3.subs({omega: 0})).simplify().subs({omega: 1})

        input_symbols = (phi, psi, x, y, v)
        self.__coeff_a = sp.lambdify(input_symbols, coeff_a, modules="numpy")
        self.__coeff_omega = sp.lambdify(input_symbols, coeff_omega, modules="numpy")
        self.__constant_term = sp.lambdify(
            input_symbols, constant_term, modules="numpy"
        )

        self.__prev_steer_angle = 0.0
        print("Safety filter initialized.")

    def __safety_filter(
        self,
        state: BicycleState,
        input_velocity: float,
        input_steer_angle: float,
        prev_velocity: float,
        prev_steer_angle: float,
    ) -> tuple[float, float]:
        """Safety filter to ensure the bicycle does not exceed certain limits.

        Args:
            state (BicycleState): The current state of the bicycle.
            input_velocity (float): The current velocity of the bicycle.
            input_steer_angle (float): The current steering angle of the bicycle.
            prev_velocity (float): The previous velocity of the bicycle.
            prev_steer_angle (float): The previous steering angle of the bicycle.

        Returns:
            tuple[float, float]: A tuple containing the filtered velocity and steering angle.
        """
        # Assume the acceleration and steer angular velocity by the difference from previous input
        a_nom = (input_velocity - prev_velocity) / self.controller_time_step
        omega_nom = (input_steer_angle - prev_steer_angle) / self.controller_time_step

        u_nom = np.array(
            [
                [a_nom],
                [omega_nom],
            ]
        )

        # argmin_u: (u-u_nom)@P@(u-u_nom)
        # st. Gu <= h
        tendency = -0.2  # -1 <= tendency <= 1
        P = np.identity(2) + np.array(
            [[0, tendency], [tendency, 0]], np.float64
        )  # Quadratic cost matrix
        q = -(2 * u_nom.T @ P).T
        coeff_inputs = (
            state.theta,
            input_steer_angle,
            state.x,
            state.y,
            state.velocity,
        )
        G = np.array(
            [
                [
                    -self.__coeff_a(*coeff_inputs),
                    -self.__coeff_omega(*coeff_inputs),
                ]
            ]
        )
        h = np.array([[self.__constant_term(*coeff_inputs)]])

        solvers.options["show_progress"] = False
        sol = solvers.qp(P=matrix(P), q=matrix(q), G=matrix(G), h=matrix(h))
        if sol["status"] != "optimal":
            print("Warning: Safety filter did not find an optimal solution.")
            velocity = input_velocity
            steer_angle = input_steer_angle
        else:
            filtered_input = np.array(sol["x"]).flatten()
            velocity = prev_velocity + self.controller_time_step * filtered_input[0]
            steer_angle = (
                prev_steer_angle + self.controller_time_step * filtered_input[1]
            )

        self.__prev_steer_angle = steer_angle
        print(
            f"state:{state.x:.2f}, {state.y:.2f}, {state.theta:.2f}, {state.velocity:.2f}"
        )
        print(f"Safety filter: velocity={velocity:.2f}, steer_angle={steer_angle:.2f}")

        return velocity, steer_angle  # Placeholder for safety filter logic


class BicycleSimResult(SimulateResult):
    def __init__(self, simulation_time: float, time_step: float):
        """Initialize the simulation result.

        Args:
            simulation_time (float): Total time of the simulation.
            time_step (float): Time step used in the simulation.
        """
        super().__init__(simulation_time, time_step)
        self.barrier_data: list[tuple[float, float, float, float]] = []

    def append_barrier_data(self, data: tuple[float, float, float, float]):
        self.barrier_data.append(data)

    def get_barrier_data(self) -> list[tuple[float, float, float, float]]:
        return self.barrier_data


class BicycleSimulator(Simulator):
    def __init__(
        self,
        model: BicycleModel,
        controller: BicycleController,
        initial_state: BicycleState,
        simulation_time: float = 10.0,
        time_step: float = 0.01,
    ):
        """Initialize the bicycle simulator.

        Args:
            model (BicycleModel): The bicycle model to simulate.
            controller (BicycleController): The controller for the simulation.
            initial_state (BicycleState): The initial state of the bicycle.
            simulation_time (float, optional): Total time of the simulation. Defaults to 10.0.
            time_step (float, optional): Time step used in the simulation. Defaults to 0.01.
        """
        super().__init__(model, controller, initial_state, simulation_time, time_step)
        self.__prev_steer_angle = 0.0

    def simulate(self) -> BicycleSimResult:
        state = self.initial_state
        t = 0.0
        control_time = 0.0
        result = BicycleSimResult(self.simulation_time, self.time_step)
        print(f"Starting simulation")
        for _ in range(self.time_steps):
            # Update control signal at specified intervals(self.control_time_step)
            if t >= control_time:
                input_signal = self.controller.control(state)
                control_time += self.control_time_step
            state = self.__RK4_step(state, input_signal)
            t += self.time_step

            # save results
            result.append_state(state)
            barrier_data = self.get_barrier_data(state, input_signal)
            result.append_barrier_data(barrier_data)
        print(f"Simulation completed.")
        return result

    def __RK4_step(
        self, state: BicycleState, input_signal: BicycleInput
    ) -> BicycleState:
        k1 = self.model.differential(state, input_signal)
        k2 = self.model.differential(state + k1 * (self.time_step / 2), input_signal)
        k3 = self.model.differential(state + k2 * (self.time_step / 2), input_signal)
        k4 = self.model.differential(state + k3 * self.time_step, input_signal)
        return state + (k1 + 2 * k2 + 2 * k3 + k4) * (self.time_step / 6)

    def get_barrier_data(
        self, state: BicycleState, input_signal: BicycleInput
    ) -> list[tuple[float, float, float, float]]:
        omega = (
            input_signal.steer - self.__prev_steer_angle
        ) / self.controller.controller_time_step
        h_inputs = (
            state.x,
            state.y,
            state.velocity,
            state.theta,
            input_signal.steer,
            input_signal.acceleration,
            omega,
        )
        h = self.controller.h(*h_inputs)  # Update safety filter state
        h1 = self.controller.h1(*h_inputs)
        h2 = self.controller.h2(*h_inputs)
        h3 = self.controller.h3(*h_inputs)

        return (h, h1, h2, h3)


class BicycleVisualizer(Visualizer):
    def __init__(
        self,
        model: BicycleModel,
        obstacle: Obstacle = None,
        fps: int = 30,
    ):
        """
        Initialize the BicycleVisualizer.

        Args:
            model (BicycleModel): The bicycle model to visualize.
            obstacle (Obstacle): The obstacle to avoid. Defaults to None.
            fps (int, optional): Frames per second for the visualization. Defaults to 30.
        """
        super().__init__(fps)
        self.model = model
        self.obstacle = obstacle

    def visualize(
        self,
        data: SimulateResult,
    ):
        """
        Visualize the simulation results.

        @param data: Simulation results (SimulateResult object).
        """
        barrier_data = data.barrier_data
        states, simulation_time, time_step = data.get_results()

        print(f"Visualizing simulation results with {len(states)} states.")

        state_x = [state.x for state in states]
        state_y = [state.y for state in states]

        # set car shape parameters
        car_length = self.model.wheelbase
        car_width = 0.5 * car_length

        # Calculate the limits for the plot
        max_x = (max(state_x) if state_x else 0) + car_length
        max_y = (max(state_y) if state_y else 0) + car_length
        min_x = (min(state_x) if state_x else 0) - car_length
        min_y = (min(state_y) if state_y else 0) - car_length

        fig, ax = plt.subplots()
        ax.set_xlim(min_x, max_x)
        ax.set_ylim(min_y, max_y)

        x_len = max_x - min_x
        y_len = max_y - min_y
        ratio = y_len / x_len  # x_len can't be zero.
        radius_inch = 12  # 12 inches for the radius of the plot. This is magic number, but i don't care
        x_size = np.sqrt(radius_inch**2 / (1 + ratio**2))
        y_size = ratio * x_size
        fig.set_size_inches(x_size, y_size)

        # Plot the trajectory as a line
        ax.plot(state_x, state_y, color="blue", linewidth=2, label="Trajectory")
        ax.grid()
        ax.legend()

        if self.obstacle is not None:
            circle = plt.Circle(
                (self.obstacle.position[0], self.obstacle.position[1]),
                self.obstacle.radius,
                color="red",
            )
            ax.add_patch(circle)
        car = plt.Rectangle(
            (0, -car_width / 2),
            car_length,
            car_width,
            angle=0,
            color="blue",
            alpha=0.5,
        )
        ax.add_patch(car)

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
            car.rotation_point = (x, y)
            car.angle = np.degrees(theta)

            return (car,)

        ani = animation.FuncAnimation(
            fig,
            animate,
            frames=len(x_fps_history),
            interval=1000 * frame_interval,
            blit=True,
        )

        print("Saving animation to bicycle_simulation.mp4")
        ani.save("bicycle_simulation.mp4", writer="ffmpeg", fps=self.fps)

        # Plot state variables over time
        figsize = (20, 8)
        rows = 4
        cols = 2
        time_points = np.arange(0, simulation_time, time_step)
        fig2, axs = plt.subplots(rows, cols, figsize=figsize, sharex=True)
        axs[0, 0].plot(time_points, [s.x for s in states], label="x")
        axs[0, 0].set_ylabel("x position")
        axs[0, 0].legend()
        axs[1, 0].plot(time_points, [s.y for s in states], label="y")
        axs[1, 0].set_ylabel("y position")
        axs[1, 0].legend()
        axs[2, 0].plot(time_points, [s.theta for s in states], label="theta")
        axs[2, 0].set_ylabel("theta (rad)")
        axs[2, 0].legend()
        axs[3, 0].plot(time_points, [s.velocity for s in states], label="velocity")
        axs[3, 0].set_ylabel("velocity")
        axs[3, 0].set_xlabel("time (s)")
        axs[3, 0].legend()
        dictionary = {
            0: "h",
            1: "h1",
            2: "h2",
            3: "h3",
        }
        for i in range(4):
            axs[i, 1].plot(
                time_points,
                [barrier_data[j][i] for j in range(len(barrier_data))],
                label=dictionary[i],
            )
            axs[i, 1].set_ylabel(dictionary[i])
            axs[i, 1].set_xlabel("time (s)")
            axs[i, 1].legend()
            axs[i, 1].grid()

        plt.tight_layout()
        plt.show()
