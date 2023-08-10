from data import User, Task, Request, generata_sytem
from genetic import genetic_algorithm, count_individual, count_schedule

# Parametre

N_USERS = 100
N_TASKS = 5
N_INPUT = 5
N_REQUESTS = 100
MIN_DATA_RATE=10
MAX_DATA_RATE = 20
MIN_INPUT_SIZE = 20
MAX_INPUT_SIZE = 100
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
POPULATION_SIZE =80
GENERATIONS = 50
MUTATION_RATE = 0.1
PROBABILITY_SKIP = 0.05


tasks, users, inputs, outputs, requests = generata_sytem(N_TASKS, N_USERS, N_INPUT,N_REQUESTS, MIN_TASK_PROCESSING_REQ,
                                                         MAX_TASK_PROCESSING_REQ, USER_COMPUTATION_CAPACITY, MIN_INPUT_SIZE,
                                                         MAX_INPUT_SIZE, MIN_OUTPUT_TIMES, MAX_OUTPUT_TIMES, MIN_ARRIVAL,
                                                         MAX_ARRIVAL, MIN_DEADLINE, MAX_DEADLINE, MEC_RADIUS)





best_schedule = genetic_algorithm(tasks, users, requests,inputs, outputs, SERVER_COMPUTATION_CAPACITY, POPULATION_SIZE, GENERATIONS,
                                  MUTATION_RATE,PROBABILITY_SKIP)

best_count = count_schedule(best_schedule, tasks, users, inputs, outputs, SERVER_COMPUTATION_CAPACITY)

# Afficher la meilleure planification trouvée
for send_time, task_schedule in best_schedule.items():
    print(f"Time Slot {send_time}:")
    for req, location in task_schedule:
        print(f"  Request {req.id}: Task {req.task}, Input {req.input_index}, Arrival {req.arrival_time}, "
              f"Deadline {req.deadline}, Execution Location: {location[0]}")
