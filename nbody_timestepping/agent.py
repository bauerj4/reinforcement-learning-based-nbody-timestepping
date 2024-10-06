from abc import ABC, abstractmethod
from enum import Enum

import numpy as np
import torch

from nbody_timestepping.particle import Particle


class Action(Enum):
    """
    Enum representing possible actions for the Q-learning agent.
    """

    KEEP_TIMESTEP = 0
    REDUCE_TIMESTEP = 1
    SEVERELY_REDUCE_TIMESTEP = 2
    INCREASE_TIMESTEP = 3
    SEVERELY_INCREASE_TIMESTEP = 4


class RLAgent(ABC):
    """
    Abstract base class for a reinforcement learning agent.
    """

    @abstractmethod
    def choose_action(self, state: tuple):
        """
        Selects an action based on the given state.

        Parameters
        ----------
        state : tuple
            The current state.

        Returns
        -------
        Action
            The action selected.
        """
        pass

    @abstractmethod
    def update_q_value(
        self, state: tuple, action: Action, reward: float, next_state: tuple
    ):
        """
        Updates the Q-value or policy for a given state-action pair.

        Parameters
        ----------
        state : tuple
            The previous state.
        action : Action
            The action taken.
        reward : float
            The reward received for the transition.
        next_state : tuple
            The new state after the transition.
        """
        pass

    @abstractmethod
    def decay_exploration(self):
        """
        Decays the exploration rate over time.
        """
        pass


class SimpleQAgent(RLAgent):
    """
    Q-learning agent for learning an optimal symplectic integrator in the N-body problem.
    Inherits from RLAgent.
    """

    def __init__(
        self,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.995,
        velocity_bins=10,
        acc_bins=10,
        dot_bins=10,
        action_enum=Action,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.velocity_bins = velocity_bins
        self.acc_bins = acc_bins
        self.dot_bins = dot_bins
        self.action_enum = action_enum

        # State space size: velocity magnitude, acceleration magnitude, and dot product
        self.state_space_size = self.velocity_bins * self.acc_bins * self.dot_bins

        # Action space size based on the Action Enum
        self.action_space_size = len(action_enum)

        # Create the q table
        self.q_table = torch.zeros((self.state_space_size, self.action_space_size))

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(list(self.action_enum))
        else:
            return self.action_enum(torch.argmax(self.q_table[state]).item())

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = torch.argmax(self.q_table[next_state])
        td_target = (
            reward + self.discount_factor * self.q_table[next_state][best_next_action]
        )
        td_error = td_target - self.q_table[state][action.value]
        self.q_table[state][action.value] += self.learning_rate * td_error

    def decay_exploration(self):
        self.exploration_rate *= self.exploration_decay

    def get_state(self, particle: Particle) -> int:
        """
        Computes the state index for a given particle.

        Parameters
        ----------
        particle : Particle
            The particle whose state needs to be computed.

        Returns
        -------
        int
            The state index for the particle in the discretized space.
        """
        # Calculate velocity and acceleration magnitudes
        velocity_magnitude = np.linalg.norm(particle.velocity)
        acceleration_magnitude = np.linalg.norm(particle.acceleration)

        # Bin the magnitudes logarithmically
        velocity_bin = min(
            int(
                np.log10(velocity_magnitude + 1e-10) / np.log10(10) * self.velocity_bins
            ),
            self.velocity_bins - 1,
        )
        acceleration_bin = min(
            int(
                np.log10(acceleration_magnitude + 1e-10) / np.log10(10) * self.acc_bins
            ),
            self.acc_bins - 1,
        )

        # Compute the dot product and normalize to the range [-1, 1]
        if velocity_magnitude > 0 and acceleration_magnitude > 0:
            dot_product = np.dot(particle.velocity, particle.acceleration) / (
                velocity_magnitude * acceleration_magnitude
            )
        else:
            dot_product = 0.0

        # Normalize dot product to a bin range from 0 to dot_product_bins
        dot_product_bin = min(
            int((dot_product + 1) / 2 * self.dot_bins), self.dot_bins - 1
        )

        # Calculate the state index
        state = (
            velocity_bin * self.acc_bins + acceleration_bin
        ) * self.dot_bins + dot_product_bin
        return state
