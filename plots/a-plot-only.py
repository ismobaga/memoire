import pickle
from collections import defaultdict

from matplotlib import pyplot as plt

from utils import tikzplotlib_fix_ncols, count_schedule, fitness_soft

plt.style.use("ggplot")

from data import User, Task, Request, generata_sytem
from genetic import genetic_algorithm, count_individual, evaluate_individual, \
    SKIP_ACTION_VALUE, two_point_crossover, uniform_crossver, selection_elites, roulette_wheel_selection, \
    crossover as crossover_one

# Parametre
from plots.params import ITERATION, N_TASKS, N_INPUT, MIN_TASK_PROCESSING_REQ, MAX_TASK_PROCESSING_REQ, MAX_DEADLINE, \
    MIN_DEADLINE, MAX_ARRIVAL, MIN_ARRIVAL, PROBABILITY_USER, PROBABILITY_SKIP, MAX_OUTPUT_TIMES, MIN_OUTPUT_TIMES, \
    MAX_INPUT_SIZE, colors, POPULATION_SIZE, GENERATIONS, MUTATION_RATE, MIN_INPUT_SIZE, N_USERS, \
    USER_COMPUTATION_CAPACITY, N_REQUESTS, MEC_RADIUS, SERVER_COMPUTATION_CAPACITY, fnames

results = {}

SKIP_ACTION_VALUE=0



def reverse_count(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return len(requests) - result

    return wrapper


non_exec_count = reverse_count(count_individual)

functions = [fitness_soft, non_exec_count]
fnamesq = {
    evaluate_individual.__name__: " Soft deadline penalty",
    fitness_soft.__name__: "Proportional Soft (0-1)",
    non_exec_count.__name__: "Hard Deadline"
}

cross_functions = [crossover_one, two_point_crossover, uniform_crossver]
# cross_functions = [uniform_crossver]
select_funtions = [selection_elites, roulette_wheel_selection]
ff = non_exec_count
for pr in select_funtions + cross_functions:
    key = ff.__name__ + pr.__name__

    name = f"{fnames[ff.__name__]}, {fnames[pr.__name__]} "
    fnames[key] = name
for correction in [50, 100]:
    for ff in functions:
        fnames[ff.__name__ + str(correction)] = f"{fnames[ff.__name__]}, {correction}\%"

ITERATION = 100


def update_results(results, result, n_users, iteration, fit_func, cross_func, select_func, correction=0):
    name = fit_func.__name__
    key = fnames[cross_func.__name__] + fnames[select_func.__name__]
    if correction != 0:
        key = fnames[name] + str(correction) + key
    else:
        key = fnames[name] + key

    # print("result", iteration, n_users, key)
    g_key = 'best_fitness_per_generation'
    if key in results[n_users]:
        results[n_users][key][g_key] = [sum(x) for x in zip(results[n_users][key][g_key], result[g_key])]
    else:
        results[n_users][key] = result

    return results


results = [{}, {}]
for u, N_USERS in enumerate([40, 100]):
    print("USERS ", N_USERS)

    for i in range(ITERATION):
        print(f"Iteration {i}")
        # Generation de données
        tasks, users, inputs, outputs, requests = generata_sytem(N_TASKS, N_USERS, N_INPUT, N_REQUESTS,
                                                                 MIN_TASK_PROCESSING_REQ,
                                                                 MAX_TASK_PROCESSING_REQ, USER_COMPUTATION_CAPACITY,
                                                                 MIN_INPUT_SIZE,
                                                                 MAX_INPUT_SIZE, MIN_OUTPUT_TIMES, MAX_OUTPUT_TIMES,
                                                                 MIN_ARRIVAL,
                                                                 MAX_ARRIVAL, MIN_DEADLINE, MAX_DEADLINE, MEC_RADIUS,
                                                                 PROBABILITY_USER)
        fit_func = non_exec_count
        select_func = selection_elites
        for crossover_func in cross_functions:
            # call genetic avec select cross hard fit
            result = genetic_algorithm(tasks, users, requests, inputs, outputs, SERVER_COMPUTATION_CAPACITY,
                                       POPULATION_SIZE,
                                       GENERATIONS,
                                       MUTATION_RATE, PROBABILITY_SKIP, draw=False, fitness_func=fit_func,
                                       selection_func=select_func, crossover_func=crossover_func)

            results = update_results(results, result, u, i, fit_func, crossover_func, select_func)

        select_func = roulette_wheel_selection
        crossover_func = uniform_crossver
        # call genetic hard fit
        result = genetic_algorithm(tasks, users, requests, inputs, outputs, SERVER_COMPUTATION_CAPACITY,
                                   POPULATION_SIZE,
                                   GENERATIONS,
                                   MUTATION_RATE, PROBABILITY_SKIP, draw=False, fitness_func=fit_func,
                                   selection_func=select_func, crossover_func=crossover_func)

        results = update_results(results, result, u, i, fit_func, crossover_func, select_func)

        fit_func = fitness_soft
        select_func = selection_elites
        # call genetic avec Propor
        result = genetic_algorithm(tasks, users, requests, inputs, outputs, SERVER_COMPUTATION_CAPACITY,
                                   POPULATION_SIZE,
                                   GENERATIONS,
                                   MUTATION_RATE, PROBABILITY_SKIP, draw=False, fitness_func=fit_func,
                                   selection_func=select_func, crossover_func=crossover_func)

        results = update_results(results, result, u, i, fit_func, crossover_func, select_func)
        for fit_func in functions:
            for apply_correction in [50, 100]:

                # call genetic with apply_correction/100
                result = genetic_algorithm(tasks, users, requests, inputs, outputs, SERVER_COMPUTATION_CAPACITY,
                                           POPULATION_SIZE,
                                           GENERATIONS,
                                           MUTATION_RATE, PROBABILITY_SKIP, draw=False, fitness_func=fit_func,
                                           selection_func=select_func, crossover_func=crossover_func, apply_correction=apply_correction/100)
                results = update_results(results, result, u, i, fit_func, crossover_func, select_func, apply_correction)


# moyenne des resultats
# print(results)
g_key = 'best_count_per_generation'
for u in [0,1]:
    # print(u)
    for key in results[u]:
        results[u][key][g_key] = [v / ITERATION for v in results[u][key][g_key]]


uid = f"g{GENERATIONS}-p{POPULATION_SIZE}-i{ITERATION}"

# ### Save resuluts
file = open(f"a-data-{uid}", 'wb')
pickle.dump(results, file)
file.close()
# ### Load results
# file = open(f"a-data-{uid}", 'rb')
# results = pickle.load(file)
# file.close()




import tikzplotlib

markers = ['x', 's', 'v', 'o', 'p', '+' ]
colors = ['b', 'g', 'r' ]

toplots= {
  "Croisement":  ["HDCR1SE", "HDCR2SE", "HDCRUSE"],
  "Selection":  ["HDCRUSE", "HDCRUSR"],
  "Fitness":  ["HDCRUSE", "PDCRUSE"],
  "Correction":  ["HDCRUSE", "HD50CRUSE", "HD100CRUSE", "PDCRUSE", "PD50CRUSE", "PD100CRUSE"]
}

for nom, courbes in toplots.items():
    fig = plt.figure(figsize=(10, 6))
    plt.xlabel("Generation")
    plt.ylabel("\% requetes executees")
    plt.title(nom)
    for u in [0,1]:
        for k, key in enumerate(courbes):
            result = results[u][key]

            markevery = list(range(0, GENERATIONS, 10))
            plt.plot(range(1, result['generations'] + 1), result[g_key],colors[u],
                     label=f"{'100' if u==1 else ''}{key}", marker=markers[k], markevery=markevery)
            fig.canvas.draw()
            plt.pause(0.1)  # pause 0.1 sec, to force a plot redraw





    plt.legend()
    plt.savefig(f"a-{nom.lower()}-{uid}.png", format='png')
    tikzplotlib_fix_ncols(fig)


    tikzplotlib.save(f"a-{nom.lower()}-{uid}.pgf")
    # plt.show()
