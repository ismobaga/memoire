import itertools
from collections import defaultdict

from matplotlib import pyplot as plt

from algos import heuristic
from utils import tikzplotlib_fix_ncols

colors = itertools.cycle(['r','g','b','c','y','m','k'])
markers = itertools.cycle(['o','s','v', 'x', '*'])
# plt.style.use("ggplot")

from data import User, Task, Request, generata_sytem
from genetic import genetic_algorithm, count_individual, count_schedule, fitness_soft, evaluate_individual, \
    SKIP_ACTION_VALUE, selection_elites, uniform_crossver, roulette_wheel_selection

# Parametre

N_USERS = 30
PROBABILITY_USER = 0.05
N_TASKS = 5
N_INPUT = 5
N_REQUESTS = 3
MIN_INPUT_SIZE = 1e3
MAX_INPUT_SIZE = 1e4
MIN_OUTPUT_TIMES = 1.5
MAX_OUTPUT_TIMES = 2
USER_COMPUTATION_CAPACITY = 1 * (10 ** 9)
MIN_TASK_PROCESSING_REQ = 100
MAX_TASK_PROCESSING_REQ = 200
MIN_ARRIVAL = 1
MAX_ARRIVAL = 40
MIN_DEADLINE = 3
MAX_DEADLINE = 8
SERVER_COMPUTATION_CAPACITY = 5 * (10 ** 9)
MEC_RADIUS = 500

# Utilisation de l'algorithme génétique
POPULATION_SIZE = 100
GENERATIONS = 100
MUTATION_RATE = 0.2
PROBABILITY_SKIP = 0.05
ITERATION =10
results = {}


def reverse_count(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return len(requests) - result

    return wrapper


non_exec_count = reverse_count(count_individual)

functions = [evaluate_individual, fitness_soft, non_exec_count]
fnames = {
    evaluate_individual.__name__: " Soft deadline penalty",
    fitness_soft.__name__: "Proportional Soft (0-1)",
    non_exec_count.__name__: "Hard Deadline",
heuristic.__name__ : "Heuristique",

}
num_req = 0
DEADLINES = list(range(MIN_DEADLINE+1, 10))
for fit_func in functions:
    results[fit_func.__name__] = [0 for v in range(len(DEADLINES))]
results[heuristic.__name__] = [0 for v in range(len(DEADLINES))]


# for u, MAX_DEADLINE in enumerate(DEADLINES):
for i in range(ITERATION):
    print("iteration", i)
    for u, MAX_DEADLINE in enumerate(DEADLINES):
        tasks, users, inputs, outputs, requests = generata_sytem(N_TASKS, N_USERS, N_INPUT, N_REQUESTS,
                                                             MIN_TASK_PROCESSING_REQ,
                                                             MAX_TASK_PROCESSING_REQ, USER_COMPUTATION_CAPACITY,
                                                             MIN_INPUT_SIZE,
                                                             MAX_INPUT_SIZE, MIN_OUTPUT_TIMES, MAX_OUTPUT_TIMES,
                                                             MIN_ARRIVAL,
                                                             MAX_ARRIVAL, MIN_DEADLINE, MAX_DEADLINE, MEC_RADIUS)
        num_req += len(requests)




        for fit_func in functions:

            result = genetic_algorithm(tasks, users, requests, inputs, outputs, SERVER_COMPUTATION_CAPACITY, POPULATION_SIZE,
                                       GENERATIONS,
                                       MUTATION_RATE, PROBABILITY_SKIP, draw=False, fitness_func=fit_func, selection_func=selection_elites, crossover_func=uniform_crossver)
            results[fit_func.__name__][u] +=  100 * result['best_count']/len(requests)

        ind, count = heuristic(tasks, users, requests, inputs, outputs, SERVER_COMPUTATION_CAPACITY)
        print("heuristique count", count)
        results[heuristic.__name__][u] += 100 * count/ len(requests)
num_req /= ITERATION
# CALCULE DE LA MOYENNE
for fit_func in functions:
    results[fit_func.__name__] = [v / ITERATION for v in results[fit_func.__name__]]

results[heuristic.__name__] = [v / ITERATION for v in results[heuristic.__name__]]

uid = f"g{GENERATIONS}-p{POPULATION_SIZE}-u{DEADLINES[0]}_{DEADLINES[-1]}-i{ITERATION}"


fig = plt.figure(figsize=(10, 6))
plt.xlabel("Max deadline")
plt.ylabel("% requetes")
plt.title(f"Pourcentage de tache execute par max deadnline avec une moyen de {num_req}")
for name, result in results.items():
    plt.plot(DEADLINES, result,
             label=f"{fnames[name]}")
    fig.canvas.draw()
    plt.pause(0.1)  # pause 0.1 sec, to force a plot redraw

plt.legend()

plt.savefig(f"deadline-{uid}.png", format='png')
tikzplotlib_fix_ncols(fig)
import tikzplotlib
tikzplotlib.save(f"deadline-{uid}.pgf")
plt.show()


