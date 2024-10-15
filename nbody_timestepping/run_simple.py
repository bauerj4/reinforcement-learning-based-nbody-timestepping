"""
Main driver for the simulation code.
"""

import sys
import json
import numpy as np
import os

from nbody_timestepping.environment import SimpleEnvironment
from nbody_timestepping.agent import SimpleQAgent
from nbody_timestepping.train import simple_rl_learning
from nbody_timestepping.particle import IntegrationMethod

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python run_simple.py <config.json> <inital_conditions>")
        sys.exit(1)

    # Load config

    with open(sys.argv[1], "r") as f:
        config = json.load(f)

    # Instantiate environment

    data_file = sys.argv[2]
    env = SimpleEnvironment()
    env.load_particles_from_gadget(data_file)

    # Run Q-Learning

    integration_method = getattr(
        IntegrationMethod, config["train"]["integration_method"]
    )
    agent = SimpleQAgent(**config["agent"])
    simple_rl_learning(
        agent,
        env,
        config["train"]["episodes"],
        config["train"]["base_timestep"],
        integration_method,
        base_steps_per_episode=config["train"]["base_steps_per_episode"],
        max_timestep=config["train"]["max_timestep"],
        min_timestep=config["train"]["min_timestep"],
        data_directory=config["train"]["data_directory"],
    )

    # Convert the PyTorch tensor to a NumPy array
    numpy_array = agent.q_table.cpu().numpy()

    # Save the NumPy array as a binary file
    np.save(os.path.join(config["train"]["data_directory"], "q_table.npy"), numpy_array)
