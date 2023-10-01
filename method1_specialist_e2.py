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


def save_results_locally(exp_fullpath, generation, fit_pop,
                         best_index, mean_fitness, std_fitness):
    """
    Saves the results for the given generation.
    """
    file_aux = open(exp_fullpath+'/results.txt', 'a')
    file_aux.write(str(generation) + ',' +
                   str(round(fit_pop[best_index], 6)) + ',' +
                   str(round(mean_fitness, 6)) + ',' +
                   str(round(std_fitness, 6)) + '\n'
                   )

    file_aux.close()


def print_generation_stats(generation, fit_pop, best_index,
                           mean_fitness, std_fitness):
    """
    Shows the stats for the given generation.
    """
    print(
        f'Generation {generation}: \
            Best Fitness: {fit_pop[best_index]}, \
            Mean Fitness: {mean_fitness}, \
            Std Fitness: {std_fitness}')


def simulation(env, x):
    """
    Plays the game with the given individual.
    """
    fitness_score, _, _, _ = env.play(pcont=x)
    return fitness_score


def evaluate(x):
    """
    Returns the fitness of the given population.
    """
    return np.array(list(map(lambda y: simulation(env, y), x)))


def mutation(numer_of_populations, population):
    """
    Performs mutation on the given population.
    MutantVector = R2 + F(R3-R4)
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
    """
    crossover_mask = np.random.rand(len(population[index])) < crossover_rate
    child = np.where(crossover_mask, mutant, population[index])

    return child


experiment_name = 'method1_specialist_e2'
n_hidden_neurons = 10

for exp in range(1, 11):
    print("Experiment: " + str(exp) + " of  10")
    print("--------------------------------------------------------")
    exp_sub_dir = 'experiment' + '_' + str(exp)
    exp_fullpath = experiment_name + '/' + exp_sub_dir

    if not os.path.exists(exp_fullpath):
        os.makedirs(exp_fullpath)

    env = Environment(experiment_name=exp_fullpath,
                      enemies=[2],
                      playermode="ai",
                      player_controller=player_controller(n_hidden_neurons),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)  # Make True to watch the game in test mode

    env.state_to_log()

    run_mode = 'train'  # train or test

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    # Differential Evolution params
    upper_bound = 1
    lower_bound = -1
    numer_of_populations = 100
    number_of_generations = 300
    mutation_scaling_factor = 0.5
    crossover_rate = 0.9

    # Initializes population
    population = np.random.uniform(lower_bound, upper_bound,
                                   (numer_of_populations, n_vars))

    # loads file with the best solution for testing
    if run_mode == 'test':

        best_solution = np.loadtxt(exp_fullpath+'/best.txt')
        print("Running best saved solution...")
        env.update_parameter('speed', 'normal')
        evaluate([best_solution])

        sys.exit(0)

    execution_time_start = time.time()

    total_experiments = 10

    # initializes population loading old solutions or generating new ones
    if not os.path.exists(exp_fullpath+'/evoman_solstate'):

        print("New Evolution...")

        fit_pop = evaluate(population)
        best_index = np.argmax(fit_pop)
        mean_fitness = np.mean(fit_pop)
        std_fitness = np.std(fit_pop)
        initial_generation = 0
        solutions = [population, fit_pop]
        env.update_solutions(solutions)

    else:

        print("Continuing Evolution...")

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
    print_generation_stats(initial_generation, fit_pop, best_index,
                           mean_fitness, std_fitness)

    # saves results for first pop
    file_aux = open(exp_fullpath+'/results.txt', 'a')
    file_aux.write('generation,best_fitness,mean_fitness,std_fitness\n')
    file_aux.close()
    save_results_locally(exp_fullpath, initial_generation, fit_pop,
                         best_index, mean_fitness, std_fitness)

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
        print_generation_stats(generation, fit_pop, best_index,
                               mean_fitness, std_fitness)

        # save results
        save_results_locally(exp_fullpath, generation, fit_pop,
                             best_index, mean_fitness, std_fitness)

        # saves generation number
        file_aux = open(exp_fullpath+'/gen.txt', 'w')
        file_aux.write(str(generation))
        file_aux.close()

        # saves file with the best solution
        np.savetxt(exp_fullpath+'/best.txt', best_solution)

        # saves simulation state
        solutions = [population, fit_pop]
        env.update_solutions(solutions)
        env.save_state()

    execution_time_end = time.time()

    print('\nExecution time: ' +
          str(
              round((execution_time_end-execution_time_start)/60)) +
          ' minutes \n')

    print('\nExecution time: ' +
          str(
              round((execution_time_end-execution_time_start))) +
          ' seconds \n')

    env.state_to_log()