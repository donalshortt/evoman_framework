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
import json

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

generations = 50
population_size = 100

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

def mutate_pops(population):

    mutants = []
    candidates = random.sample(range(50), 25)

    for candidate in candidates:
        mutant = []
        for move in range(3000):
            if random.randint(1,4) == 1:
                mutant.append(list(random.choice([1, 0]) for _ in range(5)))
            else:
                mutant.append(population[candidate][move])

        mutants.append(mutant)


    return mutants

def find_fittest_population(fittest, population):
    fittest_population = []

    for pop in range(50):
        fittest_population.append(population[fittest[pop][1]])

    return fittest_population

def print_stats(fitnesses, gains):

    first_values = list(map(lambda subarray: subarray[0] if subarray else None, fitnesses))
    valid_values = [value for value in first_values if value is not None]
    average_fitness = sum(valid_values) / len(valid_values)
    average_gain = sum(gains) / len(gains)

    print("Generation complete")
    print("Fittest pop score: " + str(fitnesses[99][0]))
    print("Least fit pop score: " + str(fitnesses[0][0]))
    print("Average fitness: " + str(average_fitness))
    print("Average gain: " + str(average_gain))

env.update_parameter('enemies', [1])

# generate the initial population of random pops 
if not os.path.exists("population.txt"):
    print("Initial population not found, generating initial population")
    subprocess.run(["python", "gen_init_pop.py"], check=True)

# you will notice im using files as variables in some cases, this is due to limitations of the framework (or my own stupidity)
if not os.path.exists("move.txt"):
    # if the file does not exist, create it
    with open("move.txt", "w") as file:
        file.write(str(0))

if not os.path.exists("refresh.txt"):
    # if the file does not exist, create it
    with open("refresh.txt", "w") as file:
        file.write(str(1))

print("\n\n<!> Starting Training <!>\n\n")

for gen in range (generations):
    fitnesses = []
    gains = []

    # make sure to tell the controller to refresh its cache
    with open("refresh.txt", "w") as file:
        file.write(str(1))

    for pop in range(population_size):
        # i need to tell the control function what pop we are currently on via a file
        with open("pop.txt", "w") as file:
            file.write(str(pop))
        
        fitness, player_life, enemy_life, game_time = env.play()
        fitnesses.append([fitness, pop])
        gains.append(player_life - enemy_life)

        # i need to make sure to set move.txt to 0 at the end of each game run
        with open("move.txt", "w") as file:
            file.write(str(0))

    
    sorted_fitnesses = sorted(fitnesses, key = lambda x: x[0])

    print_stats(sorted_fitnesses, gains)

    # i need to make sure pop.txt is reset after each subsequent generation
    with open("pop.txt", "w") as file:
        file.write(str(0))

    # find the top 50 pops and cull the rest
    # have them pair off to produce 25 children, and then make mutated copies of 25 random chads

    # send the rest to a farm upstate
    fittest = sorted_fitnesses[50:100]
   
    with open("population.txt", "r") as file:
        lines = file.readlines()

    population_data = [ast.literal_eval(line.strip()) for line in lines]

    fittest_population = find_fittest_population(fittest, population_data)
    children = mate_pops(fittest, population_data)
    mutated_population = mutate_pops(fittest_population)

    final_population = fittest_population + children + mutated_population
    final_population_list = list(final_population)

    # now i want to write the new population into the population file
    with open("population.txt", "w") as file:
       # for pop in final_population_list:
       #     print(type(pop))
       #     file.write(json.dumps(pop))
       #     file.write("\n")



        for pop in final_population_list:
            try:
                file.write(json.dumps(pop))
                file.write("\n")
            except TypeError:
                print(f"Failed to serialize: {pop}")
                #check_types(pop)

    print("\n\n<!> New Generation <!>\n\n")
