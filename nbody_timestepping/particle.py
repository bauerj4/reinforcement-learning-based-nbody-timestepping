import torch
import numpy as np

from enum import Enum
from typing import List, Any

GRAVITY = 6.6738e-8
LENGTH_UNIT_CM = 3.085678e21
MASS_UNIT_G = 1.989e43
VELOCITY_UNIT_CM_S = 1.0e5
TIME_UNIT_S = LENGTH_UNIT_CM / VELOCITY_UNIT_CM_S
INTERNAL_G = GRAVITY / LENGTH_UNIT_CM**3 * MASS_UNIT_G * TIME_UNIT_S**2


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
        self.softening = None
        acceleration = acceleration or [0.0, 0.0, 0.0]
        self.position = torch.tensor(position, dtype=torch.float32)
        self.velocity = torch.tensor(velocity, dtype=torch.float32)
        self.acceleration = torch.tensor(acceleration, dtype=torch.float32)
        self.n_acc_calculations = 0
        self.n_acc_calls = 0
        self.time_since_last_acceleration = None
        self.timestep = 1.0
        self.elapsed_kick_time = 0.0
        self.elapsed_drift_time = 0.0

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
        self.elapsed_drift_time += timestep

    def recalculate_acceleration(
        self,
        particles: List[Any],
        timestep: float = None,
        g: float = INTERNAL_G,
        force_recalculate: bool = False,
    ) -> None:
        """
        Recalculate the particle's acceleration based on the current system's state.
        This should include gravitational forces and other interactions.
        """
        self.n_acc_calls += 1
        # Do a gravity calculation
        if timestep is not None and self.time_since_last_acceleration is not None:
            self.time_since_last_acceleration += timestep
        if (
            self.time_since_last_acceleration is None
            or self.time_since_last_acceleration + timestep >= self.timestep
            or force_recalculate
        ):
            # Compute acceleration
            acceleration = torch.zeros(3)
            for p in particles:
                if p.pid == self.pid:
                    continue
                r_hat = p.position - self.position
                r_soft = (torch.norm(r_hat) ** 2 + self.softening**2) ** 0.5
                r_hat /= torch.norm(r_hat)
                # acceleration += r_hat * g * p.mass / torch.norm(r_hat) ** 3
                acceleration += r_hat * g * p.mass / r_soft**2
            acceleration = acceleration.cpu().numpy()
            self.acceleration = acceleration
            self.n_acc_calculations += 1
            self.time_since_last_acceleration = 0.0
            # print(self.position, self.velocity, self.acceleration, torch.norm(self.position))
            # print(self.time_since_last_acceleration, self.n_acc_calculations, self.acceleration, self.position)

    def kick(
        self, timestep: float, particles: List[Any], force_recalculate: bool = False
    ) -> None:
        """
        Update the velocity of the particle based on the current acceleration.

        Parameters
        ----------
        timestep : float
            The timestep for velocity update.
        particles : List[Particle]
            The particles to compute acceleration on
        force_recalculate : bool
            Overrides time accumulation scheme and calculates force anyways. Needed
            to handle half steps.
        """
        self.recalculate_acceleration(
            particles, timestep=timestep, force_recalculate=force_recalculate
        )  # Recalculate acceleration before each kick
        self.velocity += self.acceleration * timestep
        self.elapsed_kick_time += timestep
        print(
            torch.norm(torch.tensor(self.acceleration)),
            self.elapsed_drift_time,
            self.elapsed_kick_time,
        )
