import sys

from evoman.environment import Environment
from demo_controller import player_controller

import time
import numpy as np
import os

# Make False to run with visuals in test mode to see the AI play the game.
headless = True

if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


experiment_name = 'de_specialist_e8'
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

n_hidden_neurons = 10


env = Environment(experiment_name=experiment_name,
                  enemies=[8],
                  playermode="ai",
                  player_controller=player_controller(n_hidden_neurons),
                  enemymode="static",
                  level=2,
                  speed="fastest",
                  visuals=False)  # Make True to watch the game in test mode

env.state_to_log()

execution_time_start = time.time()

run_mode = 'train'
# run_mode = 'test'

# number of weights for multilayer with 10 hidden neurons
n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

# Differential Evolution params
upper_bound = 1
lower_bound = -1
numer_of_populations = 100
number_of_generations = 30
mutation_scaling_factor = 0.5
crossover_rate = 0.9

# Initializes population
population = np.random.uniform(lower_bound, upper_bound,
                               (numer_of_populations, n_vars))


def simulation(env, x):
    """
    Plays the game with the given individual.
    :param env: environment
    :param x: individual
    :return: fitness score of the individual
    """
    fitness_score, _, _, _ = env.play(pcont=x)
    return fitness_score


def evaluate(x):
    """
    Returns the fitness of the given population.
    :param x: population
    :return: list of fitness values for each individual in the population
    """
    return np.array(list(map(lambda y: simulation(env, y), x)))


def mutation(numer_of_populations, population):
    """
    Performs mutation on the given population.
    MutantVector = R2 + F(R3-R4)
    :param numer_of_populations: Population Number
    :param population: population
    """
    indices = np.random.choice(numer_of_populations, 3, replace=False)
    mutant = population[indices[0]] + (
        mutation_scaling_factor *
        (population[indices[1]] - population[indices[2]])
    )

    # Ensure that mutant is within bounds
    mutant = np.clip(mutant, lower_bound, upper_bound)

    return mutant


def crossover(index, mutant, population):
    """
    Performs crossover on the given population.
    Inherits from the parent if the crossover mask is true.
    :param i: Population Number
    """
    crossover_mask = np.random.rand(len(population[index])) < crossover_rate
    child = np.where(crossover_mask, mutant, population[index])

    return child


# loads file with the best solution for testing
if run_mode == 'test':

    best_solution = np.loadtxt(experiment_name+'/best.txt')
    print('\n RUNNING SAVED BEST SOLUTION \n')
    env.update_parameter('speed', 'normal')
    evaluate([best_solution])

    sys.exit(0)


# initializes population loading old solutions or generating new ones
if not os.path.exists(experiment_name+'/evoman_solstate'):

    print('\nNEW EVOLUTION\n')

    fit_pop = evaluate(population)
    best_index = np.argmax(fit_pop)
    mean_fitness = np.mean(fit_pop)
    std_fitness = np.std(fit_pop)
    initial_generation = 0
    solutions = [population, fit_pop]
    env.update_solutions(solutions)

else:

    print('\nCONTINUING EVOLUTION\n')

    env.load_state()
    population = env.solutions[0]
    fit_pop = env.solutions[1]

    best_index = np.argmax(fit_pop)
    mean_fitness = np.mean(fit_pop)
    std_fitness = np.std(fit_pop)

    # finds last generation number
    file_aux = open(experiment_name+'/gen.txt', 'r')
    initial_generation = int(file_aux.readline())
    file_aux.close()


# Generation 0
print(
    f'Generation {initial_generation}: \
            Best Fitness: {fit_pop[best_index]}, \
            Mean Fitness: {mean_fitness}, \
            Std Fitness: {std_fitness}')

# saves results for first pop
file_aux = open(experiment_name+'/results.txt', 'a')
file_aux.write('\n\ngen best mean std')
file_aux.write(
    '\n' + str(initial_generation) + ' ' +
    str(round(fit_pop[best_index], 6)) + ' ' +
    str(round(mean_fitness, 6)) + ' ' +
    str(round(std_fitness, 6))
)

file_aux.close()


# Evolution
for generation in range(initial_generation+1, number_of_generations):
    for i in range(numer_of_populations):
        # Mutation
        mutant = mutation(numer_of_populations, population)

        # Crossover
        child = crossover(i, mutant, population)

        # Evaluation
        fitness_child = evaluate([child])
        fitness_current = evaluate([population[i]])

        # Selection
        if fitness_child > fitness_current:
            population[i] = child

    # Log results and save best solution
    fit_pop = evaluate(population)
    best_index = np.argmax(fit_pop)
    best_solution = population[best_index]
    mean_fitness = np.mean(fit_pop)
    std_fitness = np.std(fit_pop)

    # generation number and fitness stats
    print(
        f'Generation {generation}: \
            Best Fitness: {fit_pop[best_index]}, \
            Mean Fitness: {mean_fitness}, \
            Std Fitness: {std_fitness}')

    # save results
    file_aux = open(experiment_name+'/results.txt', 'a')
    file_aux.write('\n'+str(generation) + ' ' +
                   str(round(fit_pop[best_index], 6)) + ' ' +
                   str(round(mean_fitness, 6)) + ' ' +
                   str(round(std_fitness, 6))
                   )
    file_aux.close()

    # saves generation number
    file_aux = open(experiment_name+'/gen.txt', 'w')
    file_aux.write(str(generation))
    file_aux.close()

    # saves file with the best solution
    np.savetxt(experiment_name+'/best.txt', best_solution)

    # saves simulation state
    solutions = [population, fit_pop]
    env.update_solutions(solutions)
    env.save_state()


execution_time_end = time.time()
print('\nExecution time: '+str(round((execution_time_end-execution_time_start)/60))+' minutes \n')
print('\nExecution time: '+str(round((execution_time_end-execution_time_start)))+' seconds \n')


env.state_to_log()
