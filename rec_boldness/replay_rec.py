# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:11:34 2021

@author: Ruan
"""

import numpy as np
import tensorflow as tf


class SequenceBuffer:

    def __init__(
        self,
        observation_dim,
        num_agents,
        max_seq_len = 100,
        buffer_size = 1000,  # Num episodes
        batch_size  = 32
    ):
        self.observation_dim = observation_dim
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len

        observation_buffer_shape = (buffer_size, max_seq_len, num_agents, observation_dim)
        self.observation_buffer = np.zeros(observation_buffer_shape, dtype='float32')
        
        action_buffer_shape = (buffer_size, max_seq_len, num_agents)
        self.action_buffer = np.zeros(action_buffer_shape, dtype='int32')
        self.reward_buffer = np.zeros(action_buffer_shape, dtype='float32')
        self.dones_buffer = np.zeros(action_buffer_shape, dtype='float32')
        
        mask_buffer_shape = (buffer_size, max_seq_len)
        self.mask_buffer = np.zeros(mask_buffer_shape, dtype='float32')
        
        self.counter = 0
        self.t = 0

    def can_sample_batch(self):
        return self.counter >= self.batch_size  # Cannot sample more than the batch size

    def add(
        self,
        observations,
        actions, 
        rewards,
        dones, 
    ):
        idx = self.counter % self.buffer_size  # FIFO

        self.observation_buffer[idx][self.t] = observations
        self.action_buffer[idx][self.t] = actions
        self.reward_buffer[idx][self.t] = rewards
        self.dones_buffer[idx][self.t] = dones
        self.mask_buffer[idx][self.t] = 1.0
        
        self.t += 1
        
        if all(dones):
            while self.t < self.max_seq_len:
                self.observation_buffer[idx][self.t] = np.zeros_like(observations)
                self.action_buffer[idx][self.t] = np.zeros_like(actions)
                self.reward_buffer[idx][self.t] = np.zeros_like(rewards)
                self.dones_buffer[idx][self.t] = np.zeros_like(dones)
                self.mask_buffer[idx][self.t] = 0.0
                
                self.t += 1
            
        
            self.counter += 1
            self.t = 0

    def sample(self):
        assert self.can_sample_batch()
        
        max_idx = min(self.counter, self.buffer_size)
        idxs = np.random.choice(max_idx, size=self.batch_size, replace=True)
        
        observation_batch = tf.convert_to_tensor(self.observation_buffer[idxs])
        action_batch = tf.convert_to_tensor(self.action_buffer[idxs])
        reward_batch = tf.convert_to_tensor(self.reward_buffer[idxs])
        dones_batch = tf.convert_to_tensor(self.dones_buffer[idxs])
        mask_batch = tf.convert_to_tensor(self.mask_buffer[idxs])

               

        return observation_batch, action_batch, reward_batch, dones_batch, mask_batch
