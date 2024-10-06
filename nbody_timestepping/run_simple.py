"""
Main driver for the simulation code.
"""

import sys
import json

from nbody_timestepping.environment import SimpleEnvironment
from nbody_timestepping.agent import SimpleQAgent
from nbody_timestepping.train import simple_rl_learning
from nbody_timestepping.particle import IntegrationMethod

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_gadget_hdf5.py <config.json> <inital_conditions>")
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
    )
