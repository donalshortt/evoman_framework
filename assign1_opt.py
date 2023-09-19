#######################################################################################
# EvoMan FrameWork - V1.0 2016  			                              			  #
# DEMO : perceptron neural network controller evolved by Genetic Algorithm.        	  #
#        specialist solutions for each enemy (game)                                   #
# Author: Karine Miras        			                                      		  #
# karine.smiras@gmail.com     				                              			  #
#######################################################################################

# imports framework
import sys, os

from evoman.environment import Environment
from assign1_controller import player_controller

# imports other libs
import numpy as np

experiment_name = 'controller_specialist_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
				  playermode="ai", # specifies wether controlled by human or algorithm
				  player_controller=player_controller(None),
			  	  speed="normal",
				  enemymode="static",
				  level=2,
				  visuals=True)


#for en in range(1, 9):

env.update_parameter('enemies', [1])

# load my solutions here?

sol = np.loadtxt('solutions/' + experiment_name + '1.txt')

print (env.play())
