from nbody_timestepping.agent import SimpleQAgent, Action
from nbody_timestepping.environment import SimpleEnvironment
from nbody_timestepping.particle import IntegrationMethod


def simple_rl_learning(
    agent: SimpleQAgent,
    environment: SimpleEnvironment,
    episodes: int,
    base_timestep: float,
    integration_method: IntegrationMethod,
    base_steps_per_episode=10000,
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
    """
    particles = environment.particles

    for episode in range(episodes):
        for particle in particles:
            # Get the initial state
            velocity = particle.velocity
            acceleration = particle.acceleration
            state = agent.get_state(particle)

            done = False
            while not done:
                # Agent chooses action
                action = agent.choose_action(state)

                # Update timestep based on action
                if action == Action.KEEP_TIMESTEP:
                    timestep = base_timestep
                elif action == Action.REDUCE_TIMESTEP:
                    timestep = base_timestep / 2
                elif action == Action.SEVERELY_REDUCE_TIMESTEP:
                    timestep = base_timestep / 16
                elif action == Action.INCREASE_TIMESTEP:
                    timestep = base_timestep * 2
                elif action == Action.SEVERELY_INCREASE_TIMESTEP:
                    timestep = base_timestep * 16

                # Perform integration based on the chosen method
                if integration_method == IntegrationMethod.SYMPLECTIC_ORDER_1:
                    particle.symplectic_integrator_order_1(timestep)
                elif integration_method == IntegrationMethod.SYMPLECTIC_ORDER_2:
                    particle.symplectic_integrator_order_2(timestep)
                elif integration_method == IntegrationMethod.EULER:
                    particle.euler_integrator(timestep)

                # Get the new state
                new_velocity = particle.velocity
                new_acceleration = particle.acceleration
                new_state = agent.get_state(particle)

                # Calculate reward based on the system's energy error, for example
                reward = environment.calculate_reward()

                # Update Q-values
                agent.update_q_value(state, action, reward, new_state)

                # Transition to next state
                state = new_state

            # Decay exploration rate
            agent.decay_exploration()
