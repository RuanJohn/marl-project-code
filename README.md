# marl-project-code
Multi-Agent Reinforcement Learning code implementations as part of the Advanced Topics in Reinforcement Learning reading module at the University of Cape Town 2021

All implementations are done on environments from the [ma-gym repo](https://github.com/koulanurag/ma-gym). 

The folder `feed_forward_q_networks` contains a modified version of the code from Claude Formanek's [repo](https://github.com/jcformanek/rl-starter-kit/tree/main/06-Multi-Agent-Gym) and implements shared weight Independent Q-learning agents using DQN and DDQN architectures. The `rec_q_networks` folder contains implementations of shared weight Independent Q-Learning agents using DRQN and DDRQN architectures. Agent ID is also concatentated to all individual agent observations. We also implement the VDN and Fingerprint algorithms. 