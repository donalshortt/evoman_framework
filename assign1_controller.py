# the demo_controller file contains standard controller structures for the agents.
# you can overwrite the method 'control' in your own instance of the environment
# and then use a different type of controller if you wish.
# note that the param 'controller' received by 'control' is provided through environment.play(pcont=x)
# 'controller' could contain either weights to be used in the standard controller (or other controller implemented),
# or even a full network structure (ex.: from NEAT).
from evoman.controller import Controller
import numpy as np
import random
import ast

population_cache = {}

def get_population_data(file_path):
    # TODO: refresh the cache when we modify the population
    if file_path in population_cache:
        with open("refresh.txt", "r") as file:
            refresh = int(file.read().strip())
            if refresh == 0:
                # if data is already cached, return it
                return population_cache[file_path]

    # read the population data from the file
    with open(file_path, "r") as file:
        lines = file.readlines()

    # parse and cache the population data
    population_data = [ast.literal_eval(line.strip()) for line in lines]
    population_cache[file_path] = population_data

    with open("refresh.txt", "w") as file:
        file.write(str(0))

    return population_data

# implements controller structure for player
class player_controller(Controller):
    def __init__(self, _n_hidden):
        self.n_hidden = [_n_hidden]

    def control(self, inputs, controller):

        # i want to read what to do based on our population file. so first i will read it :)
        with open ("enemy_and_experiment.txt", "r") as file:
            enemy_and_experiment = str(file.read().strip())

        population = get_population_data("population_" + str(enemy_and_experiment) + ".txt")

        # then i want to read what pop we are currently interested in
        with open("pop.txt", "r") as file:
            pop = int(file.read().strip())

        # finally, i need to know which "move" i should be performing, which is located inside the pop.
        # to know which move, i again need to read from a file
        with open("move.txt", "r+") as file:
            move_index = int(file.read().strip())

            # and then update the file for the next move
            file.seek(0)
            next_move = move_index + 1
            file.write(str(next_move))

        # now i can return which move to do!
        move = population[pop][move_index]

        return move


