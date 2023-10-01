import sys

from evoman.environment import Environment
from demo_controller import player_controller

import numpy as np
import os

# Make False to run with visuals in test mode to see the AI play the game.
headless = True

if headless:
    os.environ["SDL_VIDEODRIVER"] = "dummy"


def simulation(env, x):
    """
    Plays the game with the given individual.
    """
    fitness_score, player_energy, enemy_energy, _ = env.play(pcont=x)
    return fitness_score, player_energy, enemy_energy


exp_path = "method1_specialist_e8"  # Enemy 8
exp_subdirs = [f"experiment_{i}" for i in range(1, 11)]

for exp_subdir in exp_subdirs:
    print(f"Running Experiment {exp_subdir}...")

    # setup environment
    env = Environment(experiment_name=exp_path + '/' + exp_subdir,
                      enemies=[8],
                      playermode="ai",
                      player_controller=player_controller(10),
                      enemymode="static",
                      level=2,
                      speed="fastest",
                      visuals=False)

    exp_best_indv_gainzzz = []

    # loads file with the best solution for testing
    best_solution = np.loadtxt(exp_path + '/' + exp_subdir + '/best.txt')

    for _ in range(5):

        # run simulation on best solution
        fitness, player_energy, enemy_energy = simulation(env, best_solution)
        individual_gain = player_energy - enemy_energy

        # save individual gain
        exp_best_indv_gainzzz.append(individual_gain)

    # Convert gainzz to numpy array and calculate mean
    exp_best_indv_gainzzz = np.array(exp_best_indv_gainzzz)
    exp_best_indv_gain_mean = np.mean(exp_best_indv_gainzzz)

    print(
        f"Experiment {exp_subdir} Best solution's mean individual gain: {exp_best_indv_gain_mean}"
    )

    # save individual gain mean to file
    file_aux = open(
        exp_path + '/' + exp_subdir + '/best_indv_gain_mean.txt', 'a'
    )
    file_aux.write(str(exp_best_indv_gain_mean))
    file_aux.close()