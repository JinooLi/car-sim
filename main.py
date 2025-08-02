from simulate import Simulator
from src.bicycle_model import (
    BicycleController,
    BicycleModel,
    BicycleState,
    BicycleVisualizer,
)
from src.interface import Controller, Model, Visualizer

model: Model = BicycleModel()
controller: Controller = BicycleController(
    model, target_position=(5.0, 5.0), control_time_step=0.1
)
vis: Visualizer = BicycleVisualizer(model, fps=30)

simulator = Simulator(model, controller, simulation_time=10.0, time_step=0.001)

result = simulator.simulate(BicycleState(x=0.0, y=0.0, theta=0.0, velocity=0.0))

vis.visualize(result)
