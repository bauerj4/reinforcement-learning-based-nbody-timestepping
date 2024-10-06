import h5py

from nbody_timestepping.particle import Particle


class SimpleEnvironment:
    """
    Environment with a small number of particles, inefficient exact energy
    calculation, and no external potentials. Assumes a single particle type
    in the 0 position.
    """

    def __init__(self, G: float = 1.0):
        """
        Initializes an empty environment. Particles can be loaded from a GADGET file.

        Parameters
        ----------
        G : float, optional
            Gravitational constant, by default 1.0.
        """
        self.particles = []
        self.G = G

    def load_particles_from_gadget(self, file_path: str) -> None:
        """
        Loads particles from a GADGET HDF5 initial conditions file.

        Parameters
        ----------
        file_path : str
            The path to the GADGET HDF5 file.
        """
        with h5py.File(file_path, "r") as f:
            masses = f["PartType0/Masses"][:]
            positions = f["PartType0/Coordinates"][:]
            velocities = f["PartType0/Velocities"][:]

            for mass, pos, vel in zip(masses, positions, velocities):
                particle = Particle(mass=mass, position=pos, velocity=vel)
                self.particles.append(particle)

    def compute_total_energy(self) -> float:
        """
        Computes the total energy of the system (kinetic + potential).

        Returns
        -------
        float
            The total energy of the system.
        """
        # Kinetic energy
        total_kinetic_energy = sum(p.kinetic_energy() for p in self.particles)

        # Potential energy
        total_potential_energy = 0.0
        num_particles = len(self.particles)
        for i in range(num_particles):
            for j in range(i + 1, num_particles):
                p1, p2 = self.particles[i], self.particles[j]
                distance = np.linalg.norm(p1.position - p2.position)
                if distance > 0:  # Avoid self-interaction
                    total_potential_energy -= self.G * p1.mass * p2.mass / distance

        # Total energy is the sum of kinetic and potential energy
        return total_kinetic_energy + total_potential_energy
