from collections import defaultdict

from matplotlib import pyplot as plt

from algos import heuristic
from data import User, Task, Request, generata_sytem
from genetic import genetic_algorithm, count_individual, count_schedule, fitness_soft, evaluate_individual, \
    SKIP_ACTION_VALUE

# Parametre

N_USERS = 40
N_TASKS = 5
N_INPUT = 5
N_REQUESTS = 5
MIN_DATA_RATE = 10
MAX_DATA_RATE = 20
MIN_INPUT_SIZE = 1e3
MAX_INPUT_SIZE = 1e4
MIN_OUTPUT_TIMES = 1.5
MAX_OUTPUT_TIMES = 2.5
USER_COMPUTATION_CAPACITY = 20
MIN_TASK_PROCESSING_REQ = 10
MAX_TASK_PROCESSING_REQ = 20
MIN_ARRIVAL = 1
MAX_ARRIVAL = 40
MIN_DEADLINE = 3
MAX_DEADLINE = 8
SERVER_COMPUTATION_CAPACITY = 100
MEC_RADIUS = 100

# Utilisation de l'algorithme génétique
POPULATION_SIZE = 100
GENERATIONS = 50
MUTATION_RATE = 0.1
PROBABILITY_SKIP = 0.05



def reverse_count(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return len(requests) - result

    return wrapper

non_exec_count = reverse_count(count_individual)

functions = [evaluate_individual, fitness_soft, non_exec_count]
fnames ={
    evaluate_individual.__name__ : " Soft deadline penalty",
fitness_soft.__name__ : "Proportional Soft (0-1)",
non_exec_count.__name__ : "Non executed number",
heuristic.__name__ : "Heuristique",
}
ITERATION = 10
nreqs_max_per_time = [5, 10, 20]
mean_requests = [0]* len(nreqs_max_per_time)
# results = defaultdict(lambda : [0] * len(nreqs_max_per_time))

tasks, users, inputs, outputs, requests = generata_sytem(N_TASKS, N_USERS, N_INPUT, N_REQUESTS,
                                                                     MIN_TASK_PROCESSING_REQ,
                                                                     MAX_TASK_PROCESSING_REQ, USER_COMPUTATION_CAPACITY,
                                                                     MIN_INPUT_SIZE,
                                                                     MAX_INPUT_SIZE, MIN_OUTPUT_TIMES, MAX_OUTPUT_TIMES,
                                                                     MIN_ARRIVAL,
                                                                     MAX_ARRIVAL, MIN_DEADLINE, MAX_DEADLINE, MEC_RADIUS)




_, count = heuristic(tasks, users, requests, inputs, outputs, SERVER_COMPUTATION_CAPACITY)

print(count)


