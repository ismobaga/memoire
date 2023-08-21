from collections import defaultdict

from matplotlib import pyplot as plt

from data import User, Task, Request, generata_sytem
from genetic import genetic_algorithm, count_individual, count_schedule, SKIP_ACTION_VALUE

# Parametre

N_USERS = 5
N_TASKS = 5
N_INPUT = 5
N_REQUESTS = 5
MIN_DATA_RATE = 10
MAX_DATA_RATE = 20
MIN_INPUT_SIZE = 1e3
MAX_INPUT_SIZE = 1e5
MIN_OUTPUT_TIMES = 1.5
MAX_OUTPUT_TIMES = 2.5
USER_COMPUTATION_CAPACITY = 20
MIN_TASK_PROCESSING_REQ = 10
MAX_TASK_PROCESSING_REQ = 20
MIN_ARRIVAL = 0
MAX_ARRIVAL = 40
MIN_DEADLINE = 3
MAX_DEADLINE = 8
SERVER_COMPUTATION_CAPACITY = 100
MEC_RADIUS = 100

# Utilisation de l'algorithme génétique
POPULATION_SIZE = 100
GENERATIONS = 2000
MUTATION_RATE = 0.1
PROBABILITY_SKIP = 0.05

results = {}
for MAX_DEADLINE in range(MIN_DEADLINE+1, 8+1):
    tasks, users, inputs, outputs, requests = generata_sytem(N_TASKS, N_USERS, N_INPUT, N_REQUESTS, MIN_TASK_PROCESSING_REQ,
                                                             MAX_TASK_PROCESSING_REQ, USER_COMPUTATION_CAPACITY,
                                                             MIN_INPUT_SIZE,
                                                             MAX_INPUT_SIZE, MIN_OUTPUT_TIMES, MAX_OUTPUT_TIMES,
                                                             MIN_ARRIVAL,
                                                             MAX_ARRIVAL, MIN_DEADLINE, MAX_DEADLINE, MEC_RADIUS)

    result = genetic_algorithm(tasks, users, requests, inputs, outputs, SERVER_COMPUTATION_CAPACITY, POPULATION_SIZE,
                                      GENERATIONS,
                                      MUTATION_RATE, PROBABILITY_SKIP, draw=False)
    results[MAX_DEADLINE] = result

fig = plt.figure(figsize=(10, 6))
plt.xlabel("Generation")
plt.ylabel("Best Fitness")
plt.title("Best Fitness per Generation")
plt.grid(True)
for MAX_DEADLINE, result in results.items():
    plt.plot(range(1, result['generations'] + 1), result['best_fitness_per_generation'],
             label=f"MAX_DEADLINE {MAX_DEADLINE}, NREQUESTS {result['nrequests']}")
    fig.canvas.draw()
    print("DRAW", MAX_DEADLINE)
    plt.pause(0.1)  # pause 0.1 sec, to force a plot redraw

plt.legend()
plt.savefig("fitness.png")
plt.show()

fig = plt.figure(figsize=(10, 6))
plt.xlabel("Generation")
plt.ylabel("Best Count")
plt.title("Best Count per Generation")
for MAX_DEADLINE, result in results.items():
    plt.plot(range(1, result['generations'] + 1), result['best_count_per_generation'],
             label=f"MAX_DEADLINE {MAX_DEADLINE}, NREQUESTS {result['nrequests']}")
    fig.canvas.draw()
    print("DRAW", MAX_DEADLINE)
    plt.pause(0.1)  # pause 0.1 sec, to force a plot redraw

plt.legend()
plt.savefig("number.png")
plt.show()

# best_count = count_schedule(best_schedule, tasks, users, inputs, outputs, SERVER_COMPUTATION_CAPACITY)


individual = result['best_individual']

best_schedule = defaultdict(list)
for i, (action, req) in enumerate(zip(result['best_individual'], requests)):
    position = i
    if action is SKIP_ACTION_VALUE:
        # best_schedule[-1].append((req, individual[i]))
        continue

    location, send_time = action
    best_schedule[send_time].append((req, individual[i]))
# Afficher la meilleure planification trouvée
for send_time, task_schedule in best_schedule.items():
    print(f"Time Slot {send_time}:")
    for req, action in task_schedule:
        print(f"  Request {req.id}: Task {req.task}, Input {req.input_index}, Arrival {req.arrival_time}, "
              f"Deadline {req.deadline}, Execution Location: {action[0] if action else 'NONE'}")
