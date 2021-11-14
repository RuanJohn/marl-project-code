# -*- coding: utf-8 -*-
"""
Created on Tue Nov  9 13:07:27 2021

@author: Ruan

Some thoughts: 
    Try unque q-networks for each agent 
    Smaller Replay Buffer 
    Try doing fingerprints 
    

"""


import copy

import gym
from replay_rec import SequenceBuffer
from executor import RecurrentExecutor
from gym import Env
import sonnet as snt
from datetime import datetime
import numpy as np

# hyper parameters
BATCH_SIZE = 64
EPS_DECAY = 0.99999
LEARNING_RATE = 5e-4
UPDATE_TARGET_EVERY = 100
HIDDEN_LAYER_SIZE = 64
SOFT_UPDATE = False
LOG = True

REPLAY_BUFFER_SIZE  = 200000 
FINGER_PRINT = False
VDN_DO = False
DDQN = False
LSTM = False




def main(master_name, env_name):
    if DDQN:
        from trainer_ddqn import RecurrentTrainer
    else:
        from trainer import RecurrentTrainer
    
    
    if LOG:
        import wandb
        
        # note_string = "Batch Size: " + str(BATCH_SIZE) + \
        #     ", Eps Decay: " + str(EPS_DECAY) + \
        #         ", Buffer Size: " + str(REPLAY_BUFFER_SIZE) + \
        #             ", Hidden Layer Size: " + str(HIDDEN_LAYER_SIZE) + \
        #                 ", VDN: " + str(VDN_DO) + ", FP: " + str(FINGER_PRINT) + \
        #                     ", Soft Update: " + str(SOFT_UPDATE)
        
    
    
    env: Env = gym.make('ma_gym:' + env_name)
    
    if LOG:
        run = wandb.init(project = "MASTER-" + env_name, 
                          name = master_name)
    
    # Get env specs
    observation_dim = env.observation_space.sample()[0].shape[0]
    num_agents: int = env.n_agents
    num_actions = env.action_space._agents_action_space[0].n
    
    
    if LSTM:
        q_network = snt.DeepRNN([snt.LSTM(HIDDEN_LAYER_SIZE), snt.Linear(num_actions)])
        target_q_network = copy.deepcopy(q_network)
    
    else:
        q_network = snt.DeepRNN([snt.GRU(HIDDEN_LAYER_SIZE), snt.Linear(num_actions)])
        target_q_network = copy.deepcopy(q_network)
    
    
    if FINGER_PRINT:
        sequenceBuffer = SequenceBuffer(observation_dim + num_agents + 2, 
                                        num_agents, 
                                        batch_size = BATCH_SIZE, 
                                        buffer_size = REPLAY_BUFFER_SIZE)
    else:
        sequenceBuffer = SequenceBuffer(observation_dim + num_agents, 
                                        num_agents, 
                                        batch_size = BATCH_SIZE, 
                                        buffer_size = REPLAY_BUFFER_SIZE)
    
    executor = RecurrentExecutor(num_actions, 
                                 num_agents, 
                                 target_q_network, 
                                 sequenceBuffer, 
                                 epsilon_decay = EPS_DECAY, 
                                 finger_print = FINGER_PRINT)
    
    trainer = RecurrentTrainer(num_agents, 
                               q_network,
                               target_q_network, 
                               sequenceBuffer, 
                               batch_size=BATCH_SIZE, 
                               lr = LEARNING_RATE, 
                               target_update_period = UPDATE_TARGET_EVERY, 
                               VDN = VDN_DO, 
                               soft_update = SOFT_UPDATE)
    
    
    
    returns= []
    
    cnt = 0
    
    for episode in range(5000):
        obs = env.reset()
        
        done = [False] * num_agents
        executor.observeFirst(obs)
        ep_return = 0
        logs = {}
        
        while not all(done):
            action = executor.selectActions(obs, cnt)
            obs, reward, done, info = env.step(action)
            ep_return += sum(reward)
            
            executor.observe(obs, action, reward, done, cnt)
            cnt += 1
    
        
        loss = trainer.learn()
        if loss:
            returns.append(ep_return)
            logs["episode"] = episode
            logs["episode_return"] = ep_return
            logs["epsilon"] = executor.epsilon
            logs["loss"] = loss
            avg_window = min(cnt, 100)
            logs["avg_episode_return"] = np.mean(returns[-avg_window:])
            
            if LOG:
                run.log(logs, step = cnt)
            if episode % 500 == 0:
                print("episode:", episode, 
                      "average_ep_return:", np.mean(returns[-100:]), 
                      "epsilon:", executor.epsilon)
    
    if LOG:
        run.finish()
        
 
environments = ["Checkers-v0", "Switch2-v0", "PredatorPrey5x5-v0"]   

networks_name = ["DRQN", "DDRQN"]
finger_prints_name = ["FP", ""]
architectures_name = ["GRU", "LSTM"]
value_decomps_name = ["VDN", ""]


for environment in environments:

    for network_do in networks_name:
        name = ""
        if network_do == "DRQN":
            name += network_do
            DDQN = False
            
        else:
            name += network_do
            DDQN = True
        
        for FP in finger_prints_name:
            name = network_do
            if FP == "FP":
                
                name += " " + FP
                FINGER_PRINT = True
                REPLAY_BUFFER_SIZE = 200000
            else:
                name += " " + FP
                FINGER_PRINT = False
                REPLAY_BUFFER_SIZE = 1000
                
            
            
            for architecture in architectures_name:
                name = network_do + " " + FP
                if architecture == "GRU":
                    name += " " + architecture
                    LSTM = False
                else: 
                    name += " " + architecture
                    LSTM = True
                
                
                for value_decomp in value_decomps_name:
                    name = network_do + " " + FP + " " + architecture
                    if value_decomp == "VDN":
                        name += " " + value_decomp
                        VDN_DO = True
                    
                    else:
                        name += " " + value_decomp
                        VDN_DO = False
                    
                    print(environment + ": " + name)
                    main(name, environment)
    
 
    
 