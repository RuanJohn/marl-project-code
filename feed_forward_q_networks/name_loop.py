# -*- coding: utf-8 -*-
"""
Created on Fri Nov 12 23:49:02 2021

@author: Ruan
"""

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
                
            
            print(environment, name)
        
        
