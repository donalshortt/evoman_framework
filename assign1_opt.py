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

generations = 100
population_size = 100

# run the program 10* 3 times 

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

def generate_subarray():
    return [random.choice([1, 0]) for _ in range(5)]

def generate_main_array():
    return [generate_subarray() for _ in range(3000)]

def gen_population(population_file_name):
    arrays = [generate_main_array() for _ in range(100)]
    
    with open(population_file_name, "w") as file:
        for array in arrays:
            # Use json.dumps to create a clear, readable format
            file.write(json.dumps(array))
            file.write("\n")  # newline after each array

env.update_parameter('enemies', [1])

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

for enemy in range (3):
    for experiment in range (10):
        # i want to have a new text file generated: results + exp number + enemy name
        experiment_results = "results_" + str(enemy) + "_" + str(experiment) + ".txt"
        population_file = "population_" + str(enemy) + "_" + str(experiment) + ".txt"

        with open("enemy_and_experiment.txt", "w") as file:
            file.write(str(enemy) + "_" + str(experiment))

        gen_population(population_file)

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
                    # include the fitness, the gain & the current generation
                

                # i need to make sure to set move.txt to 0 at the end of each game run
                with open("move.txt", "w") as file:
                    file.write(str(0))

            
            sorted_fitnesses = sorted(fitnesses, key = lambda x: x[0])
            sorted_gains = sorted(gains)

            print_stats(sorted_fitnesses, gains)

            # i need to make sure pop.txt is reset after each subsequent generation
            with open("pop.txt", "w") as file:
                file.write(str(0))

            # find the top 50 pops and cull the rest
            # have them pair off to produce 25 children, and then make mutated copies of 25 random chads

            # send the rest to a farm upstate
            fittest = sorted_fitnesses[50:100]
           
            with open(population_file, "r") as file:
                lines = file.readlines()

            population_data = [ast.literal_eval(line.strip()) for line in lines]

            fittest_population = find_fittest_population(fittest, population_data)
            children = mate_pops(fittest, population_data)
            mutated_population = mutate_pops(fittest_population)

            final_population = fittest_population + children + mutated_population
            final_population_list = list(final_population)

            # now i want to write the new population into the population file
            with open(population_file, "w") as file:
                for pop in final_population_list:
                    try:
                        file.write(json.dumps(pop))
                        file.write("\n")
                    except TypeError:
                        print(f"Failed to serialize: {pop}")

            # save results!
            
            
            first_values = list(map(lambda subarray: subarray[0] if subarray else None, sorted_fitnesses))
            valid_values = [value for value in first_values if value is not None]
            average_fitness = sum(valid_values) / len(valid_values)
            average_gain = sum(gains) / len(gains)

            with open(experiment_results, "a") as file:
                file.write(
                        str(sorted_fitnesses[99][0]) + " " + 
                        str(sorted_fitnesses[0][0]) + " " + 
                        str(average_fitness) + " " + 
                        str(sorted_gains[99]) + " " + 
                        str(sorted_gains[0]) + " " + 
                        str(average_gain) + "\n"
                        )


            print("\n\n<!> New Generation <!>\n\n")
