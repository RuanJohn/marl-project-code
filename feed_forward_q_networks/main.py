import copy
import time

import gym
from replay_buffer import TransitionBuffer, Batch
from executor import Executor
from trainer import Trainer
from gym import Env
import numpy as np
import sonnet as snt
from datetime import datetime
import tensorflow as tf
import wandb


# PARAMS
N_EPISODES = 5000
TRAIN_EVERY_N_TIME_STEPS = 1
EPSILON_DECAY = 0.99999 # exponential decay
DDQN = False


def main(master_name, env_name):
    env: Env = gym.make('ma_gym:' + env_name)
    
    # Get env specs
    observation_dim = env.observation_space.sample()[0].shape[0]
    num_agents: int = env.n_agents
    num_actions = env.action_space._agents_action_space[0].n
    
    env_name = env_name[:-1] + "0"
    
    run = wandb.init(project = "MASTER-" + env_name, 
                          name = master_name)
    
    # Create networks
    q_network = snt.nets.MLP(output_sizes=(64, num_actions))
    target_q_network = copy.deepcopy(q_network)
    
    # Replay buffer
    buffer = TransitionBuffer(
        observation_dim=observation_dim,
        num_agents=num_agents, 
        buffer_size = 1000, 
        batch_size = 4
    )
    
    
    # Executor
    executor = Executor(
        q_network=target_q_network, # uses the target network
        num_agents=num_agents,
        num_actions=num_actions,
        replay_buffer=buffer,
        epsilon_decay=EPSILON_DECAY,
    )
    
    # Trainer
    trainer = Trainer(
        q_network=q_network,
        target_q_network=target_q_network,
        num_agents=num_agents,
        num_actions=num_actions,
        replay_buffer=buffer,
        ddqn = DDQN
    )
    
    t = 0 # timestep counter
    for episode in range(N_EPISODES):
        observations = env.reset()
        ep_returns = 0
        dones = [False for _ in range(num_agents)]
        logs = {}
        # Training loop
        while not all(dones):
            # Select action
            actions = executor.select_actions(observations)
    
            # Step environment
            next_observations, rewards, dones, info = env.step(actions)
    
            if t % TRAIN_EVERY_N_TIME_STEPS == 0 and buffer.can_sample_batch():
                # Periodically train
                loss = trainer.step()
                logs["loss"] = tf.reduce_sum(loss).numpy()
    
            # Create a batch from transition
            batch = Batch(
                observations=observations,
                next_observations=next_observations,
                rewards=rewards,
                actions=actions,
                dones=dones,
            )
    
            # Store transition in replay buffer
            executor.observe(batch)
    
            # Add rewards to episode return
            ep_returns += sum(rewards)
    
            # Critical!! 
            observations = next_observations
    
            # Increment timestep counter
            t += 1
    
        # Log to tensorboard
        logs["episode_return"] = ep_returns
        logs["epsilon"] = executor.epsilon
        logs["episode"] = episode
        avg_window = min(t, 100)
        logs["avg_episode_return"] = np.mean(ep_returns[-avg_window:])
        run.log(logs, step = t)
        
        if episode % 500 == 0:
            # Periodically log to terminal
            print("episode:", episode, 
                      "average_ep_return:", np.mean(ep_returns[-100:]), 
                      "epsilon:", executor.epsilon)
    
    run.finish()

networks_name = ["DQN", "DDQN"]
observability_name = ["PO", "FO"]
environments_name = ["Checkers-v0", "Switch2-v0", "PredatorPrey5x5-v0"]


for environment in environments_name:
    
    partial_environment = environment
    full_environment    = environment[:-1] + "1"

    for network_do in networks_name:
        name = ""
        if network_do == "DRQN":
            name += network_do
            DDQN = False
            
        else:
            name += network_do
            DDQN = True
        
        for observability in observability_name:
            name = network_do
            if observability == "PO":
                
                environment = partial_environment
                name += " " + observability
                
            else:
                environment = full_environment
                name += " " + observability
                
            
            print(environment + ": " + name)
            main( name, environment )

