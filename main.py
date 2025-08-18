# from simulate import Simulator
from src.bicycle_model import (
    BicycleController,
    BicycleModel,
    BicycleState,
    BicycleVisualizer,
    BicycleSimResult,
    BicycleSimulator,
    Obstacle,
)
from src.interface import Controller, Model, Visualizer
import numpy as np

model: Model = BicycleModel()


def simulate_once():
    init_state = BicycleState(x=0, y=0, theta=np.pi/4, velocity=0.0)

    obstacle = Obstacle(position=(4, 4), radius=3)

    controller: Controller = BicycleController(
        model,
        target_position=(10, 10),
        target_angle=np.pi * (3 / 12),
        controller_time_step=0.1,
        obstacle=obstacle,
        filter=True,
        steer_limit=True,
        k1=10.0,  # alpha1(a) := k1*a
        k2=10.0,  # alpha2(a) := k2*a
        k3=1,  # alpha3(a) := k3*a
    )
    vis: Visualizer = BicycleVisualizer(model, fps=30, obstacle=obstacle)

    simulator = BicycleSimulator(
        model, controller, init_state, simulation_time=15.0, time_step=0.01
    )

    result: BicycleSimResult = simulator.simulate()

    vis.visualize(result)


if __name__ == "__main__":
    simulate_once()
