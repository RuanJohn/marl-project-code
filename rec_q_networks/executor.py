# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:44:21 2021

@author: Ruan
"""

import tensorflow as tf 
import numpy as np

class RecurrentExecutor: 
    
    def __init__(self, 
                 num_actions, 
                 num_agents, 
                 q_network, 
                 replay_buffer, 
                 finger_print = False, 
                 epsilon_decay = 0.99999, 
                 epsilon_min = 0.05
                 ):
        
        self.num_actions = num_actions
        self.num_agents = num_agents 
        self.q_network = q_network
        self.replay_buffer = replay_buffer
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.epsilon = 1.0
        self.epsilon_FP = 1.0
        self.finger_print = finger_print
        
        self.observations = []
        self.states = []
        
    def observeFirst(self, observations):
        self.observations = observations
        self.states = []
        for agent in range(self.num_agents):
            self.states.append(self.q_network.initial_state(1))
        
    
    
    def observe(self, next_observations, actions, rewards, dones, train_iteration):
        
        modified_observations = []
        
        # concatenating agent ID
        for agent in range(self.num_agents):
            one_hot_agent_id = np.zeros(self.num_agents)
            one_hot_agent_id[agent] = 1
            step_num = [train_iteration/1000000]
            cur_eps = [self.epsilon_FP]
            
            if self.finger_print:
                modified_observations.append(np.concatenate([one_hot_agent_id, 
                                                             self.observations[agent],
                                                             step_num,
                                                             cur_eps])) 
            else:
                modified_observations.append(np.concatenate([one_hot_agent_id, 
                                                             self.observations[agent]])) 
        
        
        
        self.replay_buffer.add(modified_observations,
                               actions, 
                               rewards, 
                               dones)
        self.observations = next_observations
    
    
    def selectActions(self, observations, train_iteration):
        actions = []
        for agent in range(self.num_agents):
            if np.random.random() < self.epsilon:
                action = np.random.choice(self.num_actions)
            else:
                # concatenating agent ID
                one_hot_agent_id = np.zeros(self.num_agents)
                one_hot_agent_id[agent] = 1
                step_num = [train_iteration/1000000]
                cur_eps = [self.epsilon_FP]
                
                if self.finger_print:
                    modified_observation = np.concatenate([one_hot_agent_id, 
                                                           observations[agent],
                                                           step_num,
                                                           cur_eps])
                else:
                    modified_observation = np.concatenate([one_hot_agent_id, 
                                                           observations[agent]])
                    
                
                observation = tf.convert_to_tensor(modified_observation, dtype = 'float32')
                observation = tf.reshape(observation, (1, -1))
                q_values, state = self.q_network(observation, self.states[agent])
                action = tf.argmax(q_values, axis = -1).numpy()[0]
                self.states[agent] = state
                
            actions.append(action)    
        
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        self.epsilon_FP = self.epsilon_FP * self.epsilon_decay
        return actions            
            