from abc import ABC, abstractmethod
from enum import Enum


class Action(Enum):
    """
    Enum representing possible actions for the Q-learning agent.
    """

    KEEP_TIMESTEP = 0
    REDUCE_TIMESTEP = 1
    SEVERELY_REDUCE_TIMESTEP = 2


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


class SymplecticQAgent(RLAgent):
    """
    Q-learning agent for learning an optimal symplectic integrator in the N-body problem.
    Inherits from RLAgent.
    """

    def __init__(
        self,
        state_size,
        action_size,
        learning_rate=0.1,
        discount_factor=0.95,
        exploration_rate=1.0,
        exploration_decay=0.995,
        velocity_bins=10,
        acc_bins=10,
        angular_bins=16,
    ):
        super().__init__()
        self.q_table = torch.zeros(state_size + (action_size,))
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.velocity_bins = velocity_bins
        self.acc_bins = acc_bins
        self.angular_bins = angular_bins

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return np.random.choice(
                [
                    Action.KEEP_TIMESTEP,
                    Action.REDUCE_TIMESTEP,
                    Action.SEVERELY_REDUCE_TIMESTEP,
                ]
            )
        else:
            return Action(torch.argmax(self.q_table[state]).item())

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = torch.argmax(self.q_table[next_state])
        td_target = (
            reward + self.discount_factor * self.q_table[next_state][best_next_action]
        )
        td_error = td_target - self.q_table[state][action.value]
        self.q_table[state][action.value] += self.learning_rate * td_error

    def decay_exploration(self):
        self.exploration_rate *= self.exploration_decay
