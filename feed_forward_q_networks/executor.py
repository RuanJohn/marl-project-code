import tensorflow as tf

from replay_buffer import TransitionBuffer, Batch
from typing import List


class Executor:

    def __init__(
        self,
        q_network,  # Assume shared network between agents
        replay_buffer: TransitionBuffer,
        num_agents,
        num_actions,
        epsilon_start=1,
        epsilon_min=0.05,
        epsilon_decay=0.9999 # Exponential decay
    ):
        self.q_network = q_network

        self.replay_buffer = replay_buffer

        self.num_agents = num_agents
        self.num_actions = num_actions

        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def observe(
        self,
        batch: Batch, # Batch object holds: (obs, act, rew, next_obs, done)
    ):
        self.replay_buffer.add(batch)

    def select_actions(
        self,
        observations,  # [N, Obs]
    ):
        # Convert Epsilon to a tensor
        epsilon = tf.convert_to_tensor(self.epsilon, 'float32')

        actions: List[int] = []
        for agent in range(self.num_agents):
            # Convert observation to tensor
            agent_observation = tf.convert_to_tensor(observations[agent], 'float32') # [Obs]

            # Add a dummy batch dimension before passing through network
            agent_observation = tf.reshape(tensor=agent_observation, shape=(1, -1)) # [1, Obs]

            # Epsilon-greedy action selection
            agent_action = self._epsilon_greedy(agent_observation, epsilon)

            actions.append(agent_action.numpy()[0])

        # Decrement epsilon
        self._decrement_epsilon()

        return actions

    def select_greedy_actions(
        self,
        observations,  # [N, Obs]
    ):
        # Epsilon set to zero
        epsilon = tf.convert_to_tensor(0, 'float32')

        actions: List[int] = []
        for agent in range(self.num_agents):
            # Convert observation to tensor
            agent_observation = tf.convert_to_tensor(observations[agent], 'float32') # [Obs]

            # Add a dummy batch dimension before passing through network
            agent_observation = tf.reshape(tensor=agent_observation, shape=(1, -1)) # [1, Obs]

            # Epsilon-greedy action selection
            agent_action = self._epsilon_greedy(agent_observation, epsilon)

            actions.append(agent_action.numpy()[0])

        return actions

    @tf.function
    def _epsilon_greedy(
        self,
        agent_observation: tf.Tensor,
        epsilon: tf.Tensor,
    ):
        if tf.random.uniform(shape=()) < epsilon:
            agent_action = tf.random.uniform(shape=(1,), maxval=self.num_actions, dtype='int64') # explore
        else:
            q_values = self.q_network(agent_observation)
            agent_action = tf.argmax(q_values, axis=-1) # greedy

        return agent_action

    def _decrement_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
