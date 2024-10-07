import torch
import numpy as np

from enum import Enum
from typing import List, Any


class IntegrationMethod(Enum):
    """
    Enum representing the possible integration methods.

    Attributes
    ----------
    SYMPLECTIC_ORDER_1 : int
        First-order symplectic integrator (Leapfrog).
    SYMPLECTIC_ORDER_2 : int
        Second-order symplectic integrator (Leapfrog with midpoint).
    EULER : int
        Euler integration method.
    """

    SYMPLECTIC_ORDER_1 = 0
    SYMPLECTIC_ORDER_2 = 1
    EULER = 2


class Particle:
    """
    Particle class representing an N-body simulation particle.

    Parameters
    ----------
    mass : float
        Mass of the particle.
    position : list or torch.Tensor
        Initial position of the particle.
    velocity : list or torch.Tensor
        Initial velocity of the particle.
    pid : int
        The particle id
    acceleration : list or torch.Tensor
        Initial acceleration of the particle.
    """

    def __init__(
        self,
        mass: float,
        position: List[float],
        velocity: List[float],
        pid: int,
        acceleration: List[float] = None,
    ) -> None:
        self.mass = mass
        self.pid = pid
        acceleration = acceleration or [0.0, 0.0, 0.0]
        self.position = torch.tensor(position, dtype=torch.float32)
        self.velocity = torch.tensor(velocity, dtype=torch.float32)
        self.acceleration = torch.tensor(acceleration, dtype=torch.float32)
        self.n_acc_calculations = 0
        self.time_since_last_acceleration = None

    def kinetic_energy(self) -> float:
        """
        Computes the kinetic energy of the particle.

        Returns
        -------
        float
            The kinetic energy of the particle.
        """
        return 0.5 * self.mass * np.dot(self.velocity, self.velocity)

    def drift(self, timestep: float) -> None:
        """
        Update the position of the particle based on the current velocity.

        Parameters
        ----------
        timestep : float
            The timestep for position update.
        """
        self.position += self.velocity * timestep

    def recalculate_acceleration(self, particles: List[Any], g: float = 1.0) -> None:
        """
        Recalculate the particle's acceleration based on the current system's state.
        This should include gravitational forces and other interactions.
        """
        # Do a gravity calculation
        if (
            self.time_since_last_acceleration is None
            or self.time_since_last_acceleration >= self.timestep
        ):
            # Compute acceleration
            acceleration = torch.zeroes(3)
            for p in particles:
                if p.id == self.id:
                    continue
                r_hat = p.position - self.position
                acceleration += -r_hat * g * p.mass / torch.norm(r_hat) ** 3
            acceleration = acceleration.cpu().numpy().tolist()
            self.acceleration = acceleration
            self.n_acc_calculations += 1

    def kick(self, timestep: float) -> None:
        """
        Update the velocity of the particle based on the current acceleration.

        Parameters
        ----------
        timestep : float
            The timestep for velocity update.
        """
        self.recalculate_acceleration()  # Recalculate acceleration before each kick
        self.velocity += self.acceleration * timestep

    def symplectic_integrator_order_1(self, timestep: float) -> None:
        """
        Symplectic integrator of order 1 (Leapfrog method).

        Parameters
        ----------
        timestep : float
            The timestep for the integrator.
        """
        self.kick(timestep)
        self.drift(timestep)

    def symplectic_integrator_order_2(self, timestep: float) -> None:
        """
        Symplectic integrator of order 2 (Leapfrog with midpoint).

        Parameters
        ----------
        timestep : float
            The timestep for the integrator.
        """
        self.kick(timestep / 2)
        self.drift(timestep)
        self.kick(timestep / 2)

    def euler_integrator(self, timestep: float) -> None:
        """
        Euler's method for updating position and velocity.

        Parameters
        ----------
        timestep : float
            The timestep for the Euler integrator.
        """
        self.recalculate_acceleration()  # Recalculate acceleration before each step
        self.velocity += self.acceleration * timestep
        self.position += self.velocity * timestep
