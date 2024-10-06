import json
import numpy as np
import h5py
import sys
from typing import List

SOFTENING = 0.001

def read_config(config_file: str) -> dict:
    """
    Reads the JSON configuration file for the N-body problem.

    Parameters
    ----------
    config_file : str
        Path to the JSON configuration file.

    Returns
    -------
    dict
        Dictionary containing the N-body problem parameters.
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
    return config

def write_gadget_hdf5(filename: str, config: dict) -> None:
    """
    Writes the initial conditions for an N-body problem to a GADGET-4 HDF5 file.

    Parameters
    ----------
    filename : str
        Path to the output GADGET HDF5 initial conditions file.
    config : dict
        Dictionary containing the N-body problem parameters.
    """
    particles = config['particles']
    num_particles = len(particles)
    
    # Extracting time and softening length from config
    time = config.get('time', 0.0)
    softening_length = config.get('softening_length', SOFTENING)

    # Preparing data for each particle property
    masses = np.array([p['mass'] for p in particles], dtype=np.float32)
    positions = np.array([p['position'] for p in particles], dtype=np.float32)
    velocities = np.array([p['velocity'] for p in particles], dtype=np.float32)
    particle_ids = np.arange(1, num_particles + 1, dtype=np.uint32)

    # Write the data to an HDF5 file in the GADGET-4 format
    with h5py.File(filename, 'w') as f:
        # Header group
        header = f.create_group("Header")
        header.attrs["NumPart_ThisFile"] = [num_particles, 0, 0, 0, 0, 0]
        header.attrs["NumPart_Total"] = [num_particles, 0, 0, 0, 0, 0]
        header.attrs["NumPart_Total_HighWord"] = [0, 0, 0, 0, 0, 0]
        header.attrs["MassTable"] = [0, 0, 0, 0, 0, 0]
        header.attrs["Time"] = time
        header.attrs["Redshift"] = 0.0
        header.attrs["BoxSize"] = 0.0
        header.attrs["Omega0"] = 0.0
        header.attrs["OmegaLambda"] = 0.0
        header.attrs["HubbleParam"] = 1.0
        header.attrs["Flag_Sfr"] = 0
        header.attrs["Flag_Cooling"] = 0
        header.attrs["Flag_StellarAge"] = 0
        header.attrs["Flag_Metals"] = 0
        header.attrs["Flag_Feedback"] = 0
        header.attrs["Flag_DoublePrecision"] = 1
        header.attrs["SofteningLength"] = softening_length

        # Create particle data groups for positions, velocities, IDs, and masses
        part_type_0 = f.create_group("PartType0")
        part_type_0.create_dataset("Coordinates", data=positions)
        part_type_0.create_dataset("Velocities", data=velocities)
        part_type_0.create_dataset("ParticleIDs", data=particle_ids)
        part_type_0.create_dataset("Masses", data=masses)

    print(f"GADGET-4 HDF5 IC file '{filename}' generated successfully!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python generate_gadget_hdf5.py <config.json> <output_file>")
        sys.exit(1)

    config_file = sys.argv[1]
    output_file = sys.argv[2]

    config = read_config(config_file)
    write_gadget_hdf5(output_file, config)
