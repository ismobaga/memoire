import itertools
from collections import defaultdict

from matplotlib import pyplot as plt

from algos import heuristic
from utils import tikzplotlib_fix_ncols, multicast_unicast

# plt.style.use("ggplot")

from data import User, Task, Request, generata_sytem
from genetic import genetic_algorithm, count_individual, count_schedule, fitness_soft, evaluate_individual, \
    SKIP_ACTION_VALUE, roulette_wheel_selection, selection_elites, uniform_crossver

# Parametre



colors = itertools.cycle(['r','g','b','c','y','m','k'])
markers = itertools.cycle(['o','s','v', 'x', '*'])
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
GENERATIONS = 200
MUTATION_RATE = 0.2
PROBABILITY_SKIP = 0.05
ITERATION = 1
results = {}


def reverse_count(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return len(requests) - result

    return wrapper


# non_exec_count = reverse_count(count_individual)

# functions = [evaluate_individual, fitness_soft, non_exec_count]

non_exec_count = reverse_count(count_individual)
select_funtions = [roulette_wheel_selection, roulette_wheel_selection, selection_elites]
crossover_func = uniform_crossver
functions = [evaluate_individual, fitness_soft, non_exec_count]

# select_funtions = [selection_elites]
fnames = defaultdict(str)
names = [(evaluate_individual.__name__, "SD"),
         (fitness_soft.__name__, "PD"),
         (non_exec_count.__name__, "HD"),
         # (crossover_one.__name__, "CR1"),
         # (two_point_crossover.__name__, "CR2"),
         (uniform_crossver.__name__, "CRU"),
         (selection_elites.__name__, "SE"),
         (roulette_wheel_selection.__name__, "SR"),
         (heuristic.__name__ , "Heuristique")
         ]
for k, v in names:
    fnames[k] = v


for ff in functions:
    for sf in select_funtions:
        for cf in [crossover_func]:
            key = ff.__name__ + sf.__name__ + cf.__name__
            name= f"AGOR + {fnames[ff.__name__]} + {fnames[sf.__name__]} + {fnames[cf.__name__]}"
            fnames[key] = name


num_req = 0
USERS = list(range(10, 30, 10))
for fit_func, select_func in zip(functions, select_funtions):
    name = fit_func.__name__ + select_func.__name__ + crossover_func.__name__

    results[name] = [0 for v in range(len(USERS))]
results[heuristic.__name__] = [0 for v in range(len(USERS))]

for i in range(ITERATION):
    print("iteration", i)
    for u, N_USERS in enumerate(USERS):
        N_REQUESTS = N_USERS/2
        tasks, users, inputs, outputs, requests = generata_sytem(N_TASKS, N_USERS, N_INPUT, N_REQUESTS,
                                                             MIN_TASK_PROCESSING_REQ,
                                                             MAX_TASK_PROCESSING_REQ, USER_COMPUTATION_CAPACITY,
                                                             MIN_INPUT_SIZE,
                                                             MAX_INPUT_SIZE, MIN_OUTPUT_TIMES, MAX_OUTPUT_TIMES,
                                                             MIN_ARRIVAL,
                                                             MAX_ARRIVAL, MIN_DEADLINE, MAX_DEADLINE, MEC_RADIUS, PROBABILITY_USER )



        # for fit_func, select_func in zip(functions, select_funtions):
        #     num_req += len(requests)
        #     result = genetic_algorithm(tasks, users, requests, inputs, outputs, SERVER_COMPUTATION_CAPACITY, POPULATION_SIZE,
        #                                GENERATIONS,
        #                                MUTATION_RATE, PROBABILITY_SKIP, draw=False, fitness_func=fit_func, selection_func=select_func)
        #     name = fit_func.__name__ + select_func.__name__ + crossover_func.__name__
        #
        #     results[name][u] +=  100 * result['best_count']/ len(requests)

        ind, count = heuristic(tasks, users, requests, inputs, outputs, SERVER_COMPUTATION_CAPACITY)
        multi, uni = multicast_unicast(ind, requests)
        print(multi, uni)
        results[heuristic.__name__][u] += 100 * count/ len(requests)
num_req /= ITERATION*len(USERS)
# CALCULE DE LA MOYENNE
for fit_func , select_func in zip(functions, select_funtions):
    name = fit_func.__name__ + select_func.__name__ + crossover_func.__name__

    results[name] = [v / ITERATION for v in results[name]]

results[heuristic.__name__] = [v / ITERATION for v in results[heuristic.__name__]]


uid = f"g{GENERATIONS}-p{POPULATION_SIZE}-u{USERS[0]}_{USERS[-1]}-i{ITERATION}p"


fig = plt.figure(figsize=(10, 6))
plt.xlabel("n. clients")
plt.ylabel("\% requ\^{e}tes ex\'{e}cut\'{e}es")

plt.title(f"Pourcentage de requetes execute par clients")
for name, result in results.items():
    plt.plot(USERS, result,
             label=f"{fnames[name]}", marker=next(markers))
    fig.canvas.draw()
    plt.pause(0.1)  # pause 0.1 sec, to force a plot redraw

plt.legend()


plt.savefig(f"test-request-{uid}.png", format='png')
tikzplotlib_fix_ncols(fig)
import tikzplotlib
tikzplotlib.save(f"test-request-{uid}.pgf")

plt.show()

