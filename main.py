from simulate import Simulator
from src.bicycle_model import (
    BicycleController,
    BicycleModel,
    BicycleState,
    BicycleVisualizer,
)
from src.interface import Controller, Model, Visualizer
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

model: Model = BicycleModel()


def simulate_once():
    controller: Controller = BicycleController(
        model,
        target_position=(10, 10),
        target_angle=np.pi * (3 / 12),
        controller_time_step=0.01,
        filter=False,
        steer_limit=True,
    )
    vis: Visualizer = BicycleVisualizer(model, fps=30, filter=True)

    simulator = Simulator(model, controller, simulation_time=15.0, time_step=0.01)

    result = simulator.simulate(BicycleState(x=0.0, y=0.0, theta=0.0, velocity=0.0))

    vis.visualize(result)


# 병렬 처리를 위해 multiprocessing 사용
def simulate_target(args):
    i, j, target = args
    controller = BicycleController(model, target_position=target, control_time_step=0.1)
    simulator = Simulator(model, controller, simulation_time=15.0, time_step=0.01)
    result = simulator.simulate(BicycleState(x=0.0, y=0.0, theta=0.0, velocity=0.0))
    final_state = result.states[-1]
    return (i, j, 1 if final_state.velocity < 0.01 else 0)


def simulate_reachability():
    x_targets = np.arange(-10, 10.1, 0.1)
    y_targets = np.arange(-10, 10.1, 0.1)
    X, Y = np.meshgrid(x_targets, y_targets)
    success = np.zeros_like(X, dtype=int)

    targets = [
        (i, j, (X[i, j], Y[i, j])) for i in range(X.shape[0]) for j in range(X.shape[1])
    ]

    with multiprocessing.Pool() as pool:
        results = pool.map(simulate_target, targets)

    for i, j, s in results:
        success[i, j] = s

    plt.figure(figsize=(8, 8))
    plt.imshow(
        success, extent=[-10, 10, -10, 10], origin="lower", cmap="Greens", alpha=0.7
    )
    plt.xlabel("Target X")
    plt.ylabel("Target Y")
    plt.title("Reachability of Target Positions")
    plt.colorbar(label="Success (1=reachable, 0=not)")
    plt.show()


if __name__ == "__main__":
    simulate_once()
    # simulate_reachability()
