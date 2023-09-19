# the demo_controller file contains standard controller structures for the agents.
# you can overwrite the method 'control' in your own instance of the environment
# and then use a different type of controller if you wish.
# note that the param 'controller' received by 'control' is provided through environment.play(pcont=x)
# 'controller' could contain either weights to be used in the standard controller (or other controller implemented),
# or even a full network structure (ex.: from NEAT).
from evoman.controller import Controller
import numpy as np
import random

def sigmoid_activation(x):
    return 1./(1.+np.exp(-x))


# implements controller structure for player
class player_controller(Controller):
    def __init__(self, _n_hidden):
        self.n_hidden = [_n_hidden]

    def control(self, inputs, controller):
            
        # takes decisions about sprite actions

        left = random.randint(0, 1)
        right = random.randint(0, 1)
        jump = random.randint(0, 1)
        shoot = random.randint(0, 1)
        release = random.randint(0, 1)
        
        print()

        return [left, right, jump, shoot, release]


