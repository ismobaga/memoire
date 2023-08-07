import numpy as np
from collections import defaultdict


LOCAL_LOCATION = 1
SERVER_LOCATION = 2

# Fonction d'évaluation (fitness)
def evaluate_schedule(schedule, tasks, users,inputs, outputs, server_processing_capacity):
    total_penalty = 0

    for send_time, task_schedule in schedule.items():
        if send_time ==-1 :
            total_penalty += len(task_schedule)
            continue
        for req, action in task_schedule:
            task = tasks[req.task]
            user = users[req.user]
            location, send_time = action
            worst_user_req = min(task_schedule, key=lambda val: users[val[0].user].rate)
            worst_user = users[worst_user_req[0].user]
            worst_rate = worst_user.rate
            total_time  =0
            if location == LOCAL_LOCATION:
                execution_time = task.processing_requirement / user.computation_capacity
                transfert_time = inputs[req.input_index] / worst_rate
            else:
                execution_time = task.processing_requirement / server_processing_capacity
                transfert_time = outputs[req.task][req.input_index] / worst_rate
            if (send_time + execution_time + transfert_time) > (req.arrival_time + req.deadline):
                total_penalty += 1

    return total_penalty



def evaluate_individual(individual,requests, tasks, users,inputs, outputs, server_processing_capacity):
    schedule = {-1:[]}
    for i, (req, action) in enumerate(zip(requests, individual)):
        if action is None:
            schedule[-1].append((req, individual[i]))
            continue
        # print(action)
        location, send_time = action
        if send_time not in schedule:
            schedule[send_time] = []
        schedule[send_time].append((req, individual[i]))
    fitness = evaluate_schedule(schedule, tasks, users, inputs, outputs, server_processing_capacity)
    return fitness
# Opérateur de croisement (single-point crossover)
def crossover(parent1, parent2):
    crossover_point = np.random.randint(1, len(parent1))
    child1 = parent1[:crossover_point] + parent2[crossover_point:]
    child2 = parent2[:crossover_point] + parent1[crossover_point:]
    return child1, child2


# Opérateur de mutation
def mutate(individual, mutation_rate):
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        if np.random.rand() < mutation_rate:
            # send_time = np.random.randint(req.arrival_time, req.arrival_time + req.deadline)
            if mutated_individual[i] != None:
                send_time = mutated_individual[i][1]
                if mutated_individual[i][0] == LOCAL_LOCATION:
                    mutated_individual[i] = (SERVER_LOCATION, send_time)
                else:
                    mutated_individual[i] = (LOCAL_LOCATION, send_time)
    return mutated_individual


def is_valid_individual(individual, tasks, users, requests):

    send_times = defaultdict(list)
    for i, (action, req) in enumerate(zip(individual, requests)):
        if action is None:
            continue

        location, send_time = action
        send_times[send_time].append((location, req.input_index, req.task))
    for group in send_times.values():
        group_location= None
        group_input = 0
        group_task = 0
        if len(group):
            group_location, group_input, group_task = group[0]
        for element in group:
            if group_location == LOCAL_LOCATION:
                if element[1] == group_input and group_location == LOCAL_LOCATION:
                    continue
                else:
                    return  False
            else:
                if element[2] == group_task:
                    continue
                else:
                    return  False

    return True

def select_parents(population, fitness_scores, population_size):
    # Calcul des scores de fitness inversés (plus petit est meilleur)
    inverse_fitness_scores = 1 / np.array(fitness_scores)

    # Calcul des poids pour la sélection roulette
    total_inverse_fitness = np.sum(inverse_fitness_scores)
    roulette_weights = inverse_fitness_scores / total_inverse_fitness
    print(len(list(range(population_size))), len(roulette_weights), len(inverse_fitness_scores))
    # Sélection des parents (roulette)
    selected_parents_indices = np.random.choice(range(population_size), size=population_size // 2, p=roulette_weights )
    selected_parents = [population[index] for index in selected_parents_indices]

    return selected_parents


# Algorithme génétique
def genetic_algorithm(tasks, users, requests,inputs, outputs, server_processing_capacity, population_size, generations, mutation_rate, probability_skip):
    population = []
    while len(population) <population_size:
        individual = []
        for req in requests:
            if np.random.rand() < probability_skip:
                individual.append(None)
            else:
                location = np.random.choice([SERVER_LOCATION, LOCAL_LOCATION])
                send_time = np.random.randint(req.arrival_time, req.arrival_time+req.deadline)
                individual.append((location, send_time))
        # if is_valid_individual(individual, tasks, users, requests):
        population.append(individual)

    for generation in range(generations):
        print("len pop", len(population))
        # Évaluation de la population
        fitness_scores = []
        for individual in population:
            schedule = {-1:[]}
            for i, (req, action) in enumerate(zip(requests, individual)):
                if action is None:
                    schedule[-1].append((req, individual[i]))
                    continue
                # print(action)
                location, send_time = action
                if send_time not in schedule:
                    schedule[send_time] = []
                schedule[send_time].append((req, individual[i]))
            fitness = evaluate_schedule(schedule, tasks, users,inputs, outputs, server_processing_capacity)
            print(fitness)
            fitness_scores.append(fitness)

        # Sélection des parents (roullette)
        # selected_parents = np.random.choice(population, size=population_size // 2, p=1 / np.array(fitness_scores))
        # Utilisation de la fonction pour sélectionner les parents
        selected_parents = select_parents(population, fitness_scores, population_size)
        # selected_parents = select_parents(population, fitness_scores, population_size)
        print("slect", len(selected_parents))
        # elites = np.random.choice(population, size=population_size - (population_size // 2))
        # Croisement et mutation
        new_population = []
        for i in range(0, len(selected_parents)-1, 2):
            parent1 = selected_parents[i]
            parent2 = selected_parents[i + 1]
            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate)
            child2 = mutate(child2, mutation_rate)
            new_population.append(parent1)
            new_population.append(parent2)
            new_population.append(child1)
            new_population.append(child2)

        print('new', len(new_population), population_size)
        population = new_population

    # Sélection du meilleur individu
    best_individual = min(population, key=lambda ind: evaluate_individual(ind,requests, tasks, users,inputs, outputs, server_processing_capacity))
    best_schedule = {}
    for i, (req, action) in enumerate(zip(requests, best_individual)):
        if action == None:
            continue
        location, send_time = action
        if send_time not in best_schedule:
            best_schedule[send_time] = []
        best_schedule[send_time].append((req, action))

    return best_schedule


