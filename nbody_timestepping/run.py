"""
Main driver for the simulation code.
"""

import sys

from nbody_timestepping.environment import SimpleEnvironment

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_gadget_hdf5.py <config.json> <inital_conditions>")
        sys.exit(1)

    data_file = sys.argv[1]
    env = SimpleEnvironment()
    env.load_particles_from_gadget(data_file)
