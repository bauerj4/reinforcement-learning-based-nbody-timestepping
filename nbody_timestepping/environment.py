import h5py
import numpy as np
import torch

from nbody_timestepping.particle import Particle, INTERNAL_G


class SimpleEnvironment:
    """
    Environment with a small number of particles, inefficient exact energy
    calculation, and no external potentials. Assumes a single particle type
    in the 0 position.
    """

    def __init__(self, G: float = INTERNAL_G, gamma: float = 0.5):
        """
        Initializes an empty environment. Particles can be loaded from a GADGET file.

        Parameters
        ----------
        G : float, optional
            Gravitational constant, by default 1.0.
        gamma : float, optional
            The trade-off between relative energy change and
            computational cost spent in gravity calculations
        """
        self.particles = []
        self.G = G
        self.this_energy = None
        self.last_energy = None
        self.initial_energy = None
        self.gamma = gamma

    @property
    def smallest_timestep(self):
        smallest_timestep = None
        for p in self.particles:
            if smallest_timestep is None or p.timestep < smallest_timestep:
                smallest_timestep = p.timestep
        return smallest_timestep

    @property
    def biggest_timestep(self):
        biggest_timestep = None
        for p in self.particles:
            if biggest_timestep is None or p.timestep > biggest_timestep:
                biggest_timestep = p.timestep
        return biggest_timestep

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
            particle_ids = f["PartType0/ParticleIDs"][:]

            for mass, pos, vel, pid in zip(masses, positions, velocities, particle_ids):
                particle = Particle(mass=mass, position=pos, velocity=vel, pid=pid)
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
        if self.this_energy is not None:
            self.last_energy = self.this_energy

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
        te = total_kinetic_energy + total_potential_energy
        if self.this_energy is None:
            self.initial_energy = te
        self.this_energy = te
        return total_kinetic_energy + total_potential_energy

    def calculate_reward(self, n_steps: int):
        """
        Parameters
        ----------
        n_steps : int
            The number of steps the agent took whether acceleration
            was calculated or not.


        Reward = -(1 - gamma) * deltaE/E - \
            gamma * n_acc_calcs / (n_particles * total_steps)

        Returns
        -------
        reward : float
            The reward (penalty) for changes in energy and number
            of steps taken
        """
        self.compute_total_energy()
        n_acc_calculations = sum([p.n_acc_calculations for p in self.particles])
        energy_term = (
            -(1.0 - self.gamma)
            * 1e5
            * (abs((self.this_energy - self.last_energy + 1) / self.initial_energy))
        )
        steps_term = -self.gamma * (
            n_acc_calculations / (len(self.particles) * n_steps)
        )
        return energy_term + steps_term, energy_term, steps_term

    def symplectic_integrator_order_1(self, timestep: float) -> None:
        """
        Symplectic integrator of order 1 (Leapfrog method).

        Parameters
        ----------
        timestep : float
            The timestep for the integrator.
        """
        for p in self.particles:
            p.kick(timestep, self.particles)
        for p in self.particles:
            p.drift(timestep)

    def symplectic_integrator_order_2(self, timestep: float) -> None:
        """
        Symplectic integrator of order 2 (Leapfrog with midpoint).

        Parameters
        ----------
        timestep : float
            The timestep for the integrator.
        """
        # We should calculate if the total timestep will take us to a new
        # force calulation. The half steps will mess this calculation up.
        if all([p.time_since_last_acceleration is not None for p in self.particles]):
            recalculate_forces = [
                p.time_since_last_acceleration + timestep >= p.timestep
                for p in self.particles
            ]
        else:
            recalculate_forces = [True] * len(self.particles)

        for p, recalculate_force in zip(self.particles, recalculate_forces):
            p.kick(timestep / 2, self.particles, force_recalculate=recalculate_force)
        for p in self.particles:
            p.drift(timestep)
        for p, recalculate_force in zip(self.particles, recalculate_forces):
            p.kick(timestep / 2, self.particles, force_recalculate=recalculate_force)

    def euler_integrator(self, timestep: float) -> None:
        """
        Euler's method for updating position and velocity.

        Parameters
        ----------
        timestep : float
            The timestep for the Euler integrator.
        """
        for p in self.particles:
            p.recalculate_acceleration(self.particles, timestep=timestep)
        for p in self.particles:
            p.position += p.velocity * timestep
            # p.recalculate_acceleration(
            #     self.particles, timestep=timestep
            # )
            p.velocity += p.acceleration * timestep
            # print(p.position, p.velocity, p.acceleration, torch.norm(p.position))
            # import pdb; pdb.set_trace()
