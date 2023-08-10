import numpy as np
from collections import defaultdict
import random

from utils import plot

LOCAL_LOCATION = 1
SERVER_LOCATION = 2
SKIP_ACTION_VALUE = None# (-1,-1)
NON_SCHEDULE_PENALITY = 10



# Fonction d'évaluation (fitness)
def count_schedule(schedule, tasks, users,inputs, outputs, server_processing_capacity):
    executed = 0
    total_penalty = 0

    for send_time, task_schedule in schedule.items():
        if send_time ==-1 :
            continue
        if not is_valid_slot_scheduled(task_schedule) :
            total_penalty += len(task_schedule)
            continue

        # Identify the worst user in terms of transfer rate among the scheduled tasks
        worst_user_req = min(task_schedule, key=lambda val: users[val[0].user].rate)
        worst_user = users[worst_user_req[0].user]
        worst_rate = worst_user.rate
        for req, action in task_schedule:
            task = tasks[req.task]
            user = users[req.user]
            location, send_time = action


            if location == LOCAL_LOCATION:
                execution_time = task.processing_requirement / user.computation_capacity
                transfert_time = inputs[req.input_index] / worst_rate
            else:
                execution_time = task.processing_requirement / server_processing_capacity
                transfert_time = outputs[req.task][req.input_index] / worst_rate
            if (send_time + execution_time + transfert_time) <= (req.arrival_time + req.deadline):
                executed += 1

    return executed



def count_individual(individual,requests, tasks, users,inputs, outputs, server_processing_capacity):
    schedule = {-1:[]}
    for i, (req, action) in enumerate(zip(requests, individual)):
        if action == SKIP_ACTION_VALUE:
            schedule[-1].append((req, individual[i]))
            continue
        # print(action)
        location, send_time = action
        if send_time not in schedule:
            schedule[send_time] = []
        schedule[send_time].append((req, individual[i]))
    fitness = count_schedule(schedule, tasks, users, inputs, outputs, server_processing_capacity)
    return fitness


def evaluate_schedule(schedule, tasks, users,inputs, outputs, server_processing_capacity):
    total_penalty = 0

    for send_time, task_schedule in schedule.items():
        if send_time ==-1 :
            total_penalty += len(task_schedule) * NON_SCHEDULE_PENALITY
            continue
        if not is_valid_slot_scheduled(task_schedule) :
            total_penalty += len(task_schedule) * NON_SCHEDULE_PENALITY
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

            delay = send_time + execution_time + transfert_time - (req.arrival_time + req.deadline)
            penalty = max(0, delay)  # Soft deadline penalty
            total_penalty += penalty


    return total_penalty

def evaluate_individual(individual,requests, tasks, users,inputs, outputs, server_processing_capacity):
    schedule = {-1:[]}
    for i, (req, action) in enumerate(zip(requests, individual)):
        if action == SKIP_ACTION_VALUE:
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
def mutate(individual, mutation_rate, requests):
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        req = requests[i]
        if np.random.rand() < mutation_rate:
            send_time = np.random.randint(req.arrival_time, req.arrival_time + req.deadline)
            actions = [(SERVER_LOCATION, send_time), (LOCAL_LOCATION, send_time), SKIP_ACTION_VALUE]
            action_id = np.random.choice([0, 1, 2])
            mutated_individual[i] = actions[action_id] # TODO : Ajouter une faible probabilité pour le skip
            # if mutated_individual[i] != SKIP_ACTION_VALUE:
            #     # send_time = mutated_individual[i][1]
            #     if mutated_individual[i][0] == LOCAL_LOCATION:
            #         mutated_individual[i] = (SERVER_LOCATION, send_time)
            #     else:
            #         mutated_individual[i] = (LOCAL_LOCATION, send_time)
    return mutated_individual

def is_valid_slot_scheduled(group):

        if len(group):
            group_location, send_time = group[0][1] # action
            group_input = group[0][0].input_index
            group_task = group[0][0].task

            for req, action in group:
                if group_location == LOCAL_LOCATION:
                    if req.input_index == group_input and group_location == LOCAL_LOCATION:
                        continue
                    else:
                        return  False
                else:
                    if req.task == group_task:
                        continue
                    else:
                        return  False
        return True

def is_valid_individual(individual, tasks, users, requests):

    send_times = defaultdict(list)
    for i, (action, req) in enumerate(zip(individual, requests)):
        if action is SKIP_ACTION_VALUE:
            continue

        location, send_time = action
        send_times[send_time].append((location, req.input_index, req.task))
    for send_time, group in send_times.items():
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



def simple_individual_corrector(individual, requests):
    send_times = defaultdict(list)
    for i, (action, req) in enumerate(zip(individual, requests)):
        if action is SKIP_ACTION_VALUE:
            continue

        location, send_time = action
        position = i
        send_times[send_time].append((location, req.input_index, req.task, position))

    for send_time, group in send_times.items():
        group_location = None
        group_input = 0
        group_task = 0
        if len(group):
            group_location, group_input, group_task, position = group[0]
        task_fixed = False
        input_fixed =False
        while not task_fixed and not input_fixed:
            input_fixed = True
            task_fixed = True
            for element in group:
                location, input_index, task, position = element
                if location != group_location:
                    group_location = -1

                if input_index != group_input:
                    group_input = -1  # Signal that inputs are not the same
                    input_fixed = False

                if task != group_task:
                    group_task = -1  # Signal that tasks are not the same
                    task_fixed = False

            # Update the individual based on group corrections
            for i, element in enumerate(group):
                _, _, _, position = element
                if group_location != -1:
                    pass
                if group_input != -1 and group_location==-1:

                    individual[position] = (LOCAL_LOCATION, send_time)
                elif  task_fixed and group_input == -1:
                    # Ici l faut decide de  ne pas execute certaines requetes
                    skip_id = np.random.randint(len(group))
                    element_to_skip = group[skip_id]
                    _, _, _, position = element_to_skip
                    individual[position] = SKIP_ACTION_VALUE
                    group.remove(element_to_skip)
                if group_input != -1 :
                    input_fixed = True
                if group_task != -1:
                    task_fixed = True
                else:
                    # Ici l faut decide de  ne pas execute certaines requetes
                    skip_id = np.random.randint(len(group))
                    element_to_skip = group[skip_id]
                    _, _, _, position = element_to_skip
                    individual[position] = SKIP_ACTION_VALUE
                    group.remove(element_to_skip)

                # individual[i] = (location, requests[i].arrival_time)

    return individual


def correct_population(population, requests):
    corrected_population = []

    for individual in population:
        # corrected_individual = simple_individual_corrector(individual, requests)
        corrected_population.append(individual)
        # for req_idx, action in enumerate(individual):
        #     if action == SKIP_ACTION_VALUE :
        #         continue
        #     location, send_time = action
        #     corrected_location = location
        #     corrected_send_time = send_time
        #
        #     # Apply corrections here based on constraints
        #     # For example, if two requests cannot be executed together, adjust their locations or times
        #
        #     corrected_individual.append((corrected_location, corrected_send_time))



    return corrected_population

def selection(population, fitness_scores , population_size, elite_size):
    arr = np.array(fitness_scores)
    indices = arr.argsort()[-elite_size:][::-1]
    selected = []
    for i in indices:
        selected.append(population[i])

    return selected
def roulette_wheel_selection(population, fitness_scores, population_size, elite_size):
     # Calculer la somme totale des fitness
    total_fitness = sum(fitness_scores)
    inverse_fitness_scores = 1 / np.array(fitness_scores)

    # Calculer les probabilités de sélection pour chaque individu
    probabilities = [fitness / total_fitness for fitness in inverse_fitness_scores]

    # Sélectionner les individus d'élite (meilleurs individus)
    selected = []
    for _ in range(elite_size):
        # Effectuer une sélection basée sur la roulette wheel
        choice = random.choices(population, weights=probabilities, k=1)[0]
        selected.append(choice)
    return selected
def select_parents(population, fitness_scores, population_size):
    # Calcul des scores de fitness inversés (plus petit est meilleur)
    inverse_fitness_scores = 1 / np.array(fitness_scores)

    # Calcul des poids pour la sélection roulette
    total_inverse_fitness = np.sum(inverse_fitness_scores)
    roulette_weights = inverse_fitness_scores / total_inverse_fitness
    # print(len(list(range(population_size))), len(roulette_weights), len(inverse_fitness_scores))
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
                individual.append(SKIP_ACTION_VALUE)
            else:
                location = np.random.choice([SERVER_LOCATION, LOCAL_LOCATION])
                send_time = np.random.randint(req.arrival_time, req.arrival_time+req.deadline)
                individual.append((location, send_time))
        # if is_valid_individual(individual, tasks, users, requests):
        population.append(individual)

    # Initialize an empty list to store best fitness values per generation
    best_fitness_per_generation = []
    best_count_per_generation = []
    for generation in range(generations):
        # print("len pop", len(population))
        # Évaluation de la population
        fitness_scores = []
        for individual in population:
            schedule = {-1:[]}
            for i, (req, action) in enumerate(zip(requests, individual)):
                if action is SKIP_ACTION_VALUE:
                    schedule[-1].append((req, individual[i]))
                    continue
                # print(action)
                location, send_time = action
                if send_time not in schedule:
                    schedule[send_time] = []
                schedule[send_time].append((req, individual[i]))
            fitness = evaluate_schedule(schedule, tasks, users,inputs, outputs, server_processing_capacity)
            # print(fitness)
            # print(fitness)
            fitness_scores.append(fitness)

        # Append the best fitness of the current generation to the list
        best_fitness_per_generation.append(min(fitness_scores))
        best_id  = np.array(fitness_scores).argmin()
        best_individual = population[best_id]
        best_count_per_generation.append(count_individual(best_individual, requests, tasks, users, inputs, outputs, server_processing_capacity))


        # Sélection des parents (roullette)
        # selected_parents = np.random.choice(population, size=population_size // 2, p=1 / np.array(fitness_scores))
        # Utilisation de la fonction pour sélectionner les parents
        selected_parents = select_parents(population, fitness_scores, population_size)
        elites_size =  population_size // 2
        # elites = selection(population, fitness_scores, population_size,elites_size )
        # elites = roulette_wheel_selection(population, fitness_scores, population_size,elites_size )
        # selected_parents = population[elites_size:]
        # selected_parents = select_parents(population, fitness_scores, population_size)
        # print("slect", len(selected_parents))
        # elites = np.random.choice(population, size=population_size - (population_size // 2))
        # Croisement et mutation
        new_population = []
        children = []
        i  = 0
        while  len(children) < len(population):
            i += 2
            index1 = random.randint(0, elites_size - 1)
            index2 = random.randint(0, elites_size - 1)
            parent1 = population[i]
            parent2 = population[i+1]

            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate, requests)
            child2 = mutate(child2, mutation_rate, requests)

            children.append(child1)
            children.append(child2)
            children.append(parent1)
            children.append(parent2)

        new_population = children
        # print('new', len(new_population), population_size)
        corrected_population = correct_population(new_population, requests)
        population = corrected_population



    # Plotting the best fitness per generation
    plot(range(1, generations + 1), best_fitness_per_generation, 'o-', 'Generation', 'Best Fitness', 'Best Fitness per Generation')
    plot(range(1, generations + 1), best_count_per_generation, 'o-', 'Generation', 'Best N. Executed', 'Best Executed N. per Generation')

    # Sélection du meilleur individu
    best_individual = min(population, key=lambda ind: evaluate_individual(ind,requests, tasks, users,inputs, outputs, server_processing_capacity))
    best_schedule = {}
    best_fit = evaluate_individual(best_individual, requests, tasks, users, inputs, outputs, server_processing_capacity)
    # print(f"Nombre Exec {best_fit}")
    for i, (req, action) in enumerate(zip(requests, best_individual)):
        if action == SKIP_ACTION_VALUE:
            continue
        location, send_time = action
        if send_time not in best_schedule:
            best_schedule[send_time] = []
        best_schedule[send_time].append((req, action))

    return best_schedule


