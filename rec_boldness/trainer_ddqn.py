# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 14:07:11 2021

@author: Ruan
"""

import sonnet as snt
import trfl
import tensorflow as tf

class RecurrentTrainer: 
    
    def __init__(self, 
                 num_agents, 
                 q_network, 
                 target_q_network, 
                 replay_buffer, 
                 lr = 5e-4,
                 discount = 0.99, 
                 target_update_period = 100, 
                 batch_size = 32, 
                 VDN = False,
                 soft_update = False,
                 rho = 0.001):
        
        
        self.num_agents = num_agents
        self.q_network = q_network 
        self.target_q_network = target_q_network
        self.replay_buffer = replay_buffer
        self.optimizer = snt.optimizers.Adam(lr)
        self.discount = discount 
        self.target_update_period = target_update_period
        self.batch_size = batch_size
        self.trainer_steps = 0
        self.VDN = VDN
        self.rho = rho
        self.soft_update = soft_update
        
        
    
    @tf.function
    def _learn(self, 
               observations, 
               actions, 
               rewards, 
               dones, 
               mask):
        
        # change axes 
        
        
        observations = tf.transpose(observations, perm = [2, 1, 0, 3])    #[t, B, N, obs] ==> [N, t, B, obs]
        actions = tf.transpose(actions, perm = [2, 1, 0])    #[t, B, N, obs] ==> [N, t, B]
        rewards = tf.transpose(rewards, perm = [2, 1, 0])    #[t, B, N, obs] ==> [N, t, B]
        dones = tf.transpose(dones, perm = [2, 1, 0])    #[t, B, N, obs] ==> [N, t, B]
        mask = tf.transpose(mask, perm = [1, 0])    #[t, B, N, obs] ==> [N, t, B, obs]
        
        # leading dim is now actions dim, then time
        
        
        cur_observations = observations[:, 0:-1]
        next_observations = observations[:, 1: ]
        
        
        q_out = []
        q_next_out = []
               
        
        with tf.GradientTape() as tape:
            
            
            for agent in range(self.num_agents):
                
                q_values, _ = snt.static_unroll(self.q_network, cur_observations[agent],
                                                    self.q_network.initial_state(self.batch_size))
                
                q_values_choose = trfl.batched_index(q_values, actions[agent,:-1])
                
                #pass next_obs through online net 
                select_q_values, _ = snt.static_unroll(self.q_network, next_observations[agent],
                                                    self.q_network.initial_state(self.batch_size))
                
                #compute argmax for next actions 
                select_actions = tf.argmax(select_q_values, axis = -1)
                
                #pass next_obs through target net
                next_q_values, _ = snt.static_unroll(self.target_q_network, next_observations[agent],
                                                    self.target_q_network.initial_state(self.batch_size))
                
                q_values_choose_next = trfl.batched_index(next_q_values, select_actions)
                
               
                q_out.append(q_values_choose)
                q_next_out.append(q_values_choose_next)
                
            
            q_out = tf.stack(q_out)
            q_next_out = tf.stack(q_next_out)
            
            
            
            if self.VDN: 
                q_out = tf.reduce_sum(q_out, axis = 0, keepdims=True)
                q_next_out = tf.reduce_sum(q_next_out, axis = 0, keepdims=True)
                rewards = tf.reduce_sum(rewards, axis = 0, keepdims=True)
                dones = dones[0]
                dones = tf.expand_dims(dones, axis = 0)
                
            
            
            #bellman target 
            
            #comments for VDN
            #target = reward_out + self.discount * (1 - dones)
            target = rewards[:,0:-1] + self.discount * (1-dones[:,0:-1]) * q_next_out
            
            td_error = tf.stop_gradient(target) - q_out
            mask = tf.expand_dims(mask[0:-1],axis = 0)
            masked_td_error = td_error * mask
            
            loss = tf.reduce_sum(masked_td_error**2)/tf.reduce_sum(mask)  

        variables = self.q_network.trainable_variables              
        gradients = tape.gradient(loss, variables)
        self.optimizer.apply(gradients, variables)
        
        return loss
        
    
    def learn(self):
        if not self.replay_buffer.can_sample_batch():
            return False
          
        observations, actions, rewards, dones, mask = self.replay_buffer.sample()
        
        
        loss = self._learn(observations, actions, rewards, dones, mask)
        
        # Periodically update target network
        self._update_target_network()
        
        
        return loss
      
    
    def _update_target_network(self):
        
        
        """Update target network."""
        if self.trainer_steps % self.target_update_period == 0:
            online_variables = (*self.q_network.variables,)
            target_variables = (*self.target_q_network.variables,)
            
            
            if self.soft_update == True:
                # soft update 
                for src, dest in zip(online_variables, target_variables):
                    soft_update = self.rho * dest + (1 - self.rho) * src
                    
                    dest.assign(soft_update)
            
            else:
            # hard update 
                for src, dest in zip(online_variables, target_variables):
                    dest.assign(src)
            
            

        self.trainer_steps += 1
        
        
    
        
        
        
        
        