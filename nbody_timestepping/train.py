import os
import logging

from tqdm import tqdm
from copy import deepcopy

from nbody_timestepping.agent import SimpleQAgent, Action
from nbody_timestepping.environment import SimpleEnvironment
from nbody_timestepping.particle import IntegrationMethod

# Create a logger
logger = logging.getLogger(__name__)

# Set the logging level
logger.setLevel(logging.DEBUG)


def simple_rl_learning(
    agent: SimpleQAgent,
    environment: SimpleEnvironment,
    episodes: int,
    base_timestep: float,
    integration_method: IntegrationMethod,
    base_steps_per_episode: int = 10000,
    min_timestep: float = 1e-6,
    max_timestep: float = 1.0,
    data_directory: str = None,
) -> None:
    """
    Generalized reinforcement learning procedure for symplectic integrators.

    Parameters
    ----------
    agent : RLAgent
        The reinforcement learning agent.
    particles : list
        List of particle objects in the N-body simulation.
    environment : object
        The environment containing the dynamics of the N-body system.
    episodes : int
        Number of episodes for training.
    base_timestep : float
        The base timestep used in the simulation.
    integration_method : IntegrationMethod
        The integration method to be used (Symplectic order 1, 2, or Euler).
    base_steps_per_episode: int
        The number of base steps to take in the simulation
    min_timestep: float
        The minimum timestep a particle can have
    max_timestep: float
        The maximum timestep a particle can have
    data_directory: str
        Where to write simulation outputs to
    """
    initial_environment = deepcopy(environment)

    logger.info("Starting Q-Table Learning.")

    for episode in range(episodes):
        if data_directory is not None:
            energies_file = open(
                os.path.join(data_directory, f"{episode}.energies.dat"), "w"
            )
        logging.info("Doing initial energy calculation.")
        environment = deepcopy(initial_environment)
        environment.compute_total_energy()
        steps = 0
        for p in environment.particles:
            p.timestep = base_timestep

        logger.info(f"Starting episode {episode + 1} / {episodes}.")

        for step in tqdm(range(base_steps_per_episode), desc="Simulation Step: "):
            # 1. Choose action
            # 2. Advance the particles and then
            # 3. compute the RL agent properties.
            n_reduce = 0
            n_severely_reduce = 0
            n_increase = 0
            n_severely_increase = 0
            n_keep = 0

            for particle in environment.particles:
                # Get the initial state
                state = agent.get_state(particle)

                # Agent chooses action
                action = agent.choose_action(state)

                # Update timestep based on action
                if action == Action.KEEP_TIMESTEP:
                    timestep = base_timestep
                    n_keep += 1
                elif action == Action.REDUCE_TIMESTEP:
                    timestep = max(base_timestep / 2, min_timestep)
                    n_reduce += 1
                elif action == Action.SEVERELY_REDUCE_TIMESTEP:
                    timestep = max(base_timestep / 16, min_timestep)
                    n_severely_reduce += 1
                elif action == Action.INCREASE_TIMESTEP:
                    timestep = min(base_timestep * 2, max_timestep)
                    n_increase += 1
                elif action == Action.SEVERELY_INCREASE_TIMESTEP:
                    timestep = min(base_timestep * 16, max_timestep)
                    n_severely_increase += 1
                particle.timestep = timestep

            # logger.info("Choose actions:")
            # logger.info(f"Keep: {n_keep}")
            # logger.info(f"Increase: {n_increase}")
            # logger.info(f"Severely Increase: {n_severely_increase}")
            # logger.info(f"Reduce: {n_reduce}")
            # logger.info(f"Severely Reduce: {n_severely_reduce}")

            smallest_timestep = environment.smallest_timestep
            n_substeps = int(base_timestep / smallest_timestep)
            for _ in range(n_substeps):
                # Perform integration based on the chosen method
                if integration_method == IntegrationMethod.SYMPLECTIC_ORDER_1:
                    environment.symplectic_integrator_order_1(timestep)
                elif integration_method == IntegrationMethod.SYMPLECTIC_ORDER_2:
                    environment.symplectic_integrator_order_2(timestep)
                elif integration_method == IntegrationMethod.EULER:
                    environment.euler_integrator(timestep)
            # Increment steps
            steps += 1

            # Calculate reward based on the system's energy error
            reward = environment.calculate_reward(steps)

            for particle in environment.particles:
                # Get the new state
                new_state = agent.get_state(particle)

                # Update Q-values
                agent.update_q_value(state, action, reward, new_state)

                # Transition to next state
                state = new_state

            # Decay exploration rate
            agent.decay_exploration()

            # Write outputs
            if data_directory is not None:
                total_acc_calculations = sum(
                    [p.n_acc_calculations for p in environment.particles]
                )
                energies_file.write(
                    f"{steps * base_timestep}, {smallest_timestep}, {environment.this_energy}, {total_acc_calculations}\n"
                )

        if data_directory is not None:
            energies_file.close()
