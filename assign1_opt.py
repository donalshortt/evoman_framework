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
import subprocess
import ast
import random

experiment_name = 'controller_specialist_demo'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment for single objective mode (specialist)  with static enemy and ai player
env = Environment(experiment_name=experiment_name,
				  playermode="ai", # specifies wether controlled by human or algorithm
				  player_controller=player_controller(None),
			  	  speed="fastest",
				  enemymode="static",
				  level=2,
				  visuals=False)

generations = 1
population_size = 100
fitnesses = []

def mate_pops(fitnesses, population):
    
    children = []
    for x in range(0, 50, 2):
        parent_a = population[fitnesses[x][1]]
        parent_b = population[fitnesses[x + 1][1]]

        child = []
        for y in range(3000):
            if random.random() == 1:
                child.append(parent_a[y])
            else:
                child.append(parent_b[y])

        children.append(child)

    return children
        
def find_fittest_population(fittest, population):
    fittest_population = []

    for pop in range(50):
        fittest_population.append(population[fittest[pop][1]])

    return fittest_population

def mutate_pops(population):

    mutants = []
    candidates = random.sample(range(50), 25)

    for candidate in candidates:
        mutant = []
        for move in range(3000):
            if random.randint(1,4) == 1:
                mutant.append(random.choice([1, 0]) for _ in range(5))
            else:
                mutant.append(population[candidate][move])

        mutants.append(mutant)


    return mutants

def print_stats(population):
    print("look at my cool stats!")

env.update_parameter('enemies', [1])

# generate the initial population of random pops 
if not os.path.exists("population.txt"):
    subprocess.run(["python", "gen_init_pop.py"], check=True)

# you will notice im using files as variables in some cases, this is due to limitations of the framework (or my own stupidity)
if not os.path.exists("move.txt"):
    # if the file does not exist, create it
    with open("move.txt", "w") as file:
        file.write(str(0))

for gen in range (generations):
    for pop in range(population_size):
        # i need to tell the control function what pop we are currently on via a file
        with open("pop.txt", "w") as file:
            file.write(str(pop))
        
        fitness, player_life, enemy_life, game_time = env.play()
        fitnesses.append([fitness, pop])

        # i need to make sure to set move.txt to 0 at the end of each game run
        with open("move.txt", "w") as file:
            file.write(str(0))

    # i also need to make sure pop.txt is reset after each subsequent generation
    with open("pop.txt", "w") as file:
        file.write(str(0))

    # 50 chads + 25 children + 25 mutants?

    # find the top 50 pops and cull the rest
    # have them pair off to produce 25 children, and then make mutated copies of 25 random chads

    # grab the 50 chads of the population
    sorted_fitnesses = sorted(fitnesses, key = lambda x: x[0])

    # send the rest to a farm upstate
    fittest = sorted_fitnesses[50:100]

    
    with open("population.txt", "r") as file:
        lines = file.readlines()

    population_data = [ast.literal_eval(line.strip()) for line in lines]

    fittest_population = find_fittest_population(fittest, population_data)

    print("BIG LENGTH : " + str(len(fittest_population)))
    children = mate_pops(fittest, population_data)
    mutated_population = mutate_pops(fittest_population)

    final_population = fittest_population + children + mutated_population
    
    print("LEN FINAL:" + str(len(final_population)))

    print("\n\n<!> New Generation <!>\n\n")
