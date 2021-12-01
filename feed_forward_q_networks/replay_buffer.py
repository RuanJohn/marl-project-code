from typing import NamedTuple

import numpy as np
import tensorflow as tf

class Batch(NamedTuple):
    """Container for a batch of experience tuples."""
    observations: np.array
    next_observations: np.array
    rewards: np.array
    actions: np.array
    dones: np.array


class TransitionBuffer:

    def __init__(
        self,
        observation_dim,
        num_agents,
        buffer_size=5000,  # Num episodes
        batch_size=32
    ):
        self.observation_dim = observation_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size

        observation_buffer_shape = (buffer_size, num_agents, observation_dim)
        self.observation_buffer = np.zeros(observation_buffer_shape, dtype='float32')
        self.next_observation_buffer = np.zeros(observation_buffer_shape, dtype='float32')

        action_buffer_shape = (buffer_size, num_agents)
        self.action_buffer = np.zeros(action_buffer_shape, dtype='int32')
        self.reward_buffer = np.zeros(action_buffer_shape, dtype='float32')
        self.dones_buffer = np.zeros(action_buffer_shape, dtype='float32')

        self.counter = 0

    def can_sample_batch(self):
        return self.counter >= self.batch_size  # Cannot sample more than the batch size

    def add(
        self,
        batch: Batch,
    ):
        idx = self.counter % self.buffer_size  # FIFO

        self.observation_buffer[idx] = batch.observations
        self.next_observation_buffer[idx] = batch.next_observations
        self.action_buffer[idx] = batch.actions
        self.reward_buffer[idx] = batch.rewards
        self.dones_buffer[idx] = batch.dones

        self.counter += 1

    def sample(self):
        assert self.can_sample_batch()

        max_idx = min(self.counter, self.buffer_size)
        idxs = np.random.choice(max_idx, size=self.batch_size, replace=True)

        observation_batch = tf.convert_to_tensor(self.observation_buffer[idxs])
        next_observation_batch = tf.convert_to_tensor(self.next_observation_buffer[idxs])
        action_batch = tf.convert_to_tensor(self.action_buffer[idxs])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[idxs])
        dones_batch = tf.convert_to_tensor(self.dones_buffer[idxs])

        batch = Batch(
            observations=observation_batch,
            next_observations=next_observation_batch,
            actions=action_batch,
            rewards=reward_batch,
            dones=dones_batch,
        )

        return batch
