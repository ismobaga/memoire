import numpy as np
from collections import defaultdict
import random

from utils import plot, individual_conflict_corrector, is_valid_slot_scheduled, form_send_group

LOCAL_LOCATION = 1
SERVER_LOCATION = 2
SKIP_ACTION_VALUE = None  # (-1,-1)
NON_SCHEDULE_PENALITY = 10


# Fonction d'évaluation (fitness)
def count_schedule(schedule, tasks, users, inputs, outputs, server_processing_capacity):
    executed = 0
    total_penalty = 0

    for send_time, task_schedule in schedule.items():
        if send_time == -1:
            total_penalty += len(task_schedule)
            continue
        if not is_valid_slot_scheduled(task_schedule):
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


def count_individual(individual, requests, tasks, users, inputs, outputs, server_processing_capacity):
    schedule = {-1: []}
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


def fitness_soft(individual, requests, tasks, users, inputs, outputs, server_processing_capacity):
    groups = form_send_group(individual, requests, True)
    PENALITY = 1
    total_penalty = 0
    for send_time, group in groups.items():
        if send_time == -1:
            total_penalty += len(group) * PENALITY
        else:
            worst_user_req = min(group, key=lambda val: users[requests[val[3]].user].rate)
            worst_user = users[requests[worst_user_req[3]].user]
            worst_rate = worst_user.rate
            for location, input_index, task, position in group:
                req = requests[position]
                task = tasks[req.task]
                user = users[req.user]
                if location == LOCAL_LOCATION:
                    execution_time = task.processing_requirement / user.computation_capacity
                    transfert_time = inputs[req.input_index] / worst_rate
                else:
                    execution_time = task.processing_requirement / server_processing_capacity
                    transfert_time = outputs[req.task][req.input_index] / worst_rate

                total = send_time + execution_time + transfert_time
                delay = total - (req.arrival_time + req.deadline)
                penalty = max(0, delay)  # Soft deadline penalty

                if penalty >= 4:
                    penalty = 2
                # elif penalty >= 3:
                #     penalty = 0.75
                # elif penalty >= 2:
                #     penalty = 0.5
                elif penalty > 0:
                    penalty = 1
                total_penalty += penalty
    return total_penalty


def evaluate_schedule(schedule, tasks, users, inputs, outputs, server_processing_capacity):
    total_penalty = 0

    for send_time, task_schedule in schedule.items():
        if send_time == -1:
            total_penalty += len(task_schedule) * NON_SCHEDULE_PENALITY
            continue
        if not is_valid_slot_scheduled(task_schedule):
            total_penalty += len(task_schedule) * NON_SCHEDULE_PENALITY
            continue
        for req, action in task_schedule:
            task = tasks[req.task]
            user = users[req.user]
            location, send_time = action
            worst_user_req = min(task_schedule, key=lambda val: users[val[0].user].rate)
            worst_user = users[worst_user_req[0].user]
            worst_rate = worst_user.rate
            total_time = 0
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


def evaluate_individual(individual, requests, tasks, users, inputs, outputs, server_processing_capacity):
    schedule = {-1: []}
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


def two_point_crossover(parent1, parent2):
    # child = [[None] * len(parent1[0])] * 2
    pivot1 = np.random.randint(1, len(parent1) - 2)
    pivot2 = np.random.randint(pivot1 + 1, len(parent1))
    pivot1, pivot2 = min(pivot1, pivot2), max(pivot1, pivot2)
    child1 = parent1[:pivot1] + parent2[pivot1:pivot2] + parent1[pivot2:]
    child2 = parent2[:pivot1] + parent1[pivot1:pivot2] + parent2[pivot2:]
    return child1, child2


def uniform_crossver(parent1, parent2):
    child1, child2 = [], []
    for gene1, gene2 in zip(parent1, parent2):
        if random.random() < 0.5:
            child1.append(gene1)
            child2.append(gene2)
        else:
            child1.append(gene2)
            child2.append(gene1)
    return child1, child2


# Opérateur de mutation
def mutate(individual, mutation_rate, requests):
    mutated_individual = individual.copy()
    for i in range(len(mutated_individual)):
        req = requests[i]
        if np.random.rand() < mutation_rate:
            send_time = np.random.randint(req.arrival_time, req.arrival_time + req.deadline)
            actions = [(SERVER_LOCATION, send_time), (LOCAL_LOCATION, send_time), SKIP_ACTION_VALUE]
            action_id = np.random.choice([0, 1, 2], p=[0.45, 0.45, 0.1])
            mutated_individual[i] = actions[action_id]  # TODO : Ajouter une faible probabilité pour le skip
    return mutated_individual


def correct_population(population, requests):
    corrected_population = []

    for individual in population:
        corrected_individual = individual_conflict_corrector(individual, requests)
        corrected_population.append(corrected_individual)

    return corrected_population


def selection(population, fitness_scores, elite_size):
    arr = np.array(fitness_scores)
    indices = arr.argsort()[-elite_size:][::-1]
    selected = []
    for i in indices:
        selected.append(population[i])

    return selected


def selection_elites(population, fitness_values, elite_size):
    # Sélection des élites de la population actuelle
    elites_indices = np.argsort(fitness_values)[:int(elite_size)]
    elites = []
    for indice in elites_indices:
        elites.append(population[indice])
    return elites

def roulette_selection(population, fitness_values, elite_size):
    fs = fitness_values
    sum_fs = sum(fs)
    max_fs = max(fs)
    min_fs = min(fs)
    p = random.random()*sum_fs
    t = max_fs + min_fs
    choosen = population[0]
    selected = []
    while len(selected) < elite_size:
        for i, choosen in enumerate(population):
            p -= (t - fitness_values[i])
            if p < 0:
                selected.append(choosen)
                break
    return selected
def roulette_wheel_selection_bac(population, fitness_values, elite_size):
    selected = []
    for _ in range(elite_size):
        mf = min(fitness_values) + 1
        maxf = max(fitness_values) + 1
        fitness_values = maxf -  np.array(fitness_values)
        fitness_values = list(fitness_values)




        # Inversion des valeurs de fitness (pour minimisation)
        # inverted_fitness = [1 / f for f in fitness_values]

        # Calcul de la somme totale des valeurs de fitness inversées
        total_fitness = sum(fitness_values)

        # Calcul des probabilités de sélection pour chaque individu
        selection_probabilities = [f / total_fitness for f in fitness_values]
        print("fitness",fitness_values)
        print("probabilite selection max",selection_probabilities)
        print("max proba", max(selection_probabilities))
        print("min proba", min(selection_probabilities))

        # Génération d'un nombre aléatoire entre 0 et 1
        random_number = random.uniform(0, 1)

        # Sélection de l'individu en utilisant la sélection par roulette
        cumulative_probability = 0
        selected_individual = None

        for i, probability in enumerate(selection_probabilities):
            cumulative_probability += probability
            if random_number <= cumulative_probability:
                selected_individual = i
                break
        selected.append(population[selected_individual])
    return selected

def roulette_wheel_selection(population, fitness_scores, elite_size):
    sum_fs = sum(fitness_scores)
    max_fs = max(fitness_scores)
    min_fs = min(fitness_scores)
    # Calculer la somme totale des fitness

    fitness_scores = 4**(max_fs - np.array(fitness_scores, dtype=np.float64))
    total_fitness = sum(fitness_scores)
    import sys
    # print("sys.maxsize", sys.maxsize)
    # print("total", total_fitness)
    # total_fitness = sum(inverse_fitness_scores)

    # Calculer les probabilités de sélection pour chaque individu
    probabilities = [ fitness / total_fitness for fitness in fitness_scores]
    # print(len(population[0]))
    # print("fitness",fitness_scores)
    # print("proba",probabilities)
    # print("sum",sum(probabilities))

    # Sélectionner les individus d'élite (meilleurs individus)
    selected = []
    for _ in range(elite_size):
        # Effectuer une sélection basée sur la roulette wheel
        choice = random.choices(population, weights=probabilities, k=1)[0]
        selected.append(choice)
    return selected


def selection(population, fitness_values):
    total_fitness = sum(fitness_values)

    # Normalize fitness values
    normalized_fitness = [total_fitness / fit for fit in fitness_values]

    # Selection using Roulette Wheel Selection
    selected_population = []
    for _ in range(len(population)):
        random_value = random.uniform(0, total_fitness)
        cumulative_sum = 0
        for index, norm_fit in enumerate(normalized_fitness):
            cumulative_sum += norm_fit
            if cumulative_sum >= random_value:
                selected_population.append(population[index])
                break


def select_parents(population, size):
    return random.sample(population, size)


def crossover_population(population, size, parents=None, crossover_func=crossover):
    if parents is None:
        parents = select_parents(population, size)

    # Effectue le croisement
    children = []
    for i in range(0, len(parents), 2):
        parent1 = parents[i]
        parent2 = parents[i + 1]
        child1, child2 = crossover_func(parent1, parent2)
        children.append(child1)
        children.append(child2)
    return children


def mutate_population(population, mutation_rate, requests):
    for i in range(len(population)):
        if random.random() < mutation_rate:
            population[i] = mutate(population[i], mutation_rate, requests)
    return population


def get_next_generation(population, tasks, users, requests, inputs, outputs, server_processing_capacity, elite_size,
                        fitness_func, selection_func, crossover_func, croisement_rate, mutation_rate, apply_correction):
    fitness_scores = [fitness_func(individual, requests, tasks, users, inputs, outputs, server_processing_capacity) for
                      individual in population]

    elites = selection_func(population, fitness_scores, elite_size)
    # parents_size = len(population) * croisement_rate
    parents_size = len(population) - elite_size
    # parents = selection_elites(population, fitness_scores, parents_size)
    parents = selection_func(population, fitness_scores, parents_size)

    offspring = crossover_population(population, parents_size, parents, crossover_func)

    children = mutate_population(offspring, mutation_rate, requests)
    new_population = elites + children
    # corrected_population = correct_population(new_population, requests)
    corrected_population = new_population
    apply = False
    if apply_correction == 1:
        apply = True
    if 0 < apply_correction < 1 and random.random() > apply_correction:
        apply = True
    if apply:
        corrected_population = correct_population(new_population, requests)

    return corrected_population


def create_individual(requests, probability_skip):
    individual = []
    for req in requests:
        if np.random.rand() < probability_skip:
            individual.append(SKIP_ACTION_VALUE)
        else:
            location = np.random.choice([SERVER_LOCATION, LOCAL_LOCATION])
            send_time = np.random.randint(req.arrival_time, req.arrival_time + req.deadline)
            individual.append((location, send_time))
    return individual


def create_initial_population(requests, population_size, probability_skip):
    population = [create_individual(requests, probability_skip) for i in range(population_size)]
    return population


# apply_correction = 1 : veut dire que la correction est appliqué a tous les individu
# apply_correction < 1 : veu dire que avec une probabilité de apply_correction corriger
def genetic_algorithm(tasks, users, requests, inputs, outputs, server_processing_capacity, population_size, generations,
                      mutation_rate, probability_skip, draw=False, fitness_func=None, selection_func=None,
                      crossover_func=None, apply_correction=1):
    def reverse_count(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return len(requests) - result

        return wrapper

    if fitness_func == None:
        fitness_func = evaluate_individual
        wrapped_for_count_inv = reverse_count(count_individual)
        fitness_func = wrapped_for_count_inv
    if selection_func == None:
        selection_func = selection_elites
    if crossover_func == None:
        crossover_func = uniform_crossver


    croisement_rate = 0.2
    elite_size = int(croisement_rate * (population_size))  # int(population_size * 0.2)
    fitnesses = []
    population = create_initial_population(requests, population_size, probability_skip)
    population = correct_population(population, requests)
    best_fitness_per_generation = []
    best_count_per_generation = []
    best_percent_per_generation = []
    for generation in range(generations):

        fitness_scores = []
        for individual in population:
            fitness = fitness_func(individual, requests, tasks, users, inputs, outputs, server_processing_capacity)

            fitness_scores.append(fitness)
        min_fit = min(fitness_scores)
        # Append the best fitness of the current generation to the list
        best_fitness_per_generation.append(min_fit)
        best_id = np.array(fitness_scores).argmin()
        best_individual = population[best_id]
        best_individual = population[best_id]
        # print("Gneneration", generation, "min fit", min_fit, fitness_scores[best_id], best_individual)
        count = count_individual(best_individual, requests, tasks, users, inputs, outputs, server_processing_capacity)
        best_count_per_generation.append(100 * count / len(requests))
        # best_percent_per_generation.append(100 *
        #                                    count_individual(best_individual, requests, tasks, users, inputs, outputs,
        #                                                     server_processing_capacity) / len(requests))

        if generation % 5 == 0 and draw:
            fitness_scores = [
                fitness_func(individual, requests, tasks, users, inputs, outputs, server_processing_capacity)
                for
                individual in population]

            ef = min(fitness_scores)
            fitnesses += [[generation, ef]]
            population = sorted(population, key=lambda x: fitness_func(x, requests, tasks, users, inputs, outputs,
                                                                       server_processing_capacity))

        if (generation + 1) == generations:
            apply_correction = 1
        population = get_next_generation(population, tasks, users, requests, inputs, outputs,
                                         server_processing_capacity, elite_size, fitness_func, selection_func,
                                         crossover_func, croisement_rate,
                                         mutation_rate, apply_correction=apply_correction)
    if draw:
        fitness = np.array(fitnesses)
        print(fitness)
        plot(fitness[:, 0], fitness[:, 1], 'o-', title=fitness_func.__name__)
    best_individual = min(population, key=lambda ind: fitness_func(ind, requests, tasks, users, inputs, outputs,
                                                                   server_processing_capacity))

    data = {
        'generations': generations,
        'best_fitness_per_generation': best_fitness_per_generation,
        'best_count_per_generation': best_count_per_generation,

        'nrequests': len(requests),
        # 'population': population,
        'best_individual': best_individual,
        'best_count': count_individual(best_individual, requests, tasks, users, inputs, outputs,
                                       server_processing_capacity)
    }

    return data


# Algorithme génétique
def genetic_algorithm2(tasks, users, requests, inputs, outputs, server_processing_capacity, population_size,
                       generations,
                       mutation_rate, probability_skip, draw=False, fitness_func=None):
    def reverse_count(func):
        def wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            return len(requests) - result

        return wrapper

    if fitness_func == None:
        fitness_func = evaluate_individual
        wrapped_for_count_inv = reverse_count(count_individual)
        fitness_func = wrapped_for_count_inv
    population = []
    while len(population) < population_size:
        individual = create_individual(requests, probability_skip)
        # if is_valid_individual(individual, tasks, users, requests):
        population.append(individual)
    population = correct_population(population, requests)

    # Initialize an empty list to store best fitness values per generation
    best_fitness_per_generation = []
    best_count_per_generation = []
    for generation in range(generations):
        # print("len pop", len(population))
        # Évaluation de la population

        population = sorted(population, key=lambda x: fitness_func(x, requests, tasks, users, inputs, outputs,
                                                                   server_processing_capacity))
        fitness_scores = []
        for individual in population:
            fitness = fitness_func(individual, requests, tasks, users, inputs, outputs, server_processing_capacity)

            fitness_scores.append(fitness)
        min_fit = min(fitness_scores)
        # Append the best fitness of the current generation to the list
        best_fitness_per_generation.append(min_fit)
        best_id = np.array(fitness_scores).argmin()
        best_individual = population[best_id]
        # print("Gneneration", generation, "min fit", min_fit, fitness_scores[best_id], best_individual)
        best_count_per_generation.append(
            count_individual(best_individual, requests, tasks, users, inputs, outputs, server_processing_capacity))

        ELITISM_RATIO = 0.30
        # Find indices of top individuals (elitism)
        elites_size = int(ELITISM_RATIO * len(population))
        elite_indices = sorted(range(len(fitness_scores)), key=lambda i: fitness_scores[i])[:elites_size]

        # Create a new generation with elites and offspring
        new_population = [population[i] for i in elite_indices]
        while len(new_population) < len(population):
            index1 = random.randint(0, elites_size - 1)
            index2 = random.randint(0, elites_size - 1)
            parent1 = population[index1]
            parent2 = population[index2]

            child1, child2 = crossover(parent1, parent2)
            child1 = mutate(child1, mutation_rate, requests)
            child2 = mutate(child2, mutation_rate, requests)

            new_population.append(child1)
            new_population.append(child2)

        corrected_population = correct_population(new_population, requests)
        population = corrected_population

    if draw:
        # Plotting the best fitness per generation
        plot(range(1, generations + 1), best_fitness_per_generation, 'o-', 'Generation', 'Best Fitness',
             'Best Fitness per Generation')
        plot(range(1, generations + 1), best_count_per_generation, 'o-', 'Generation', 'Best N. Executed',
             'Best Executed N. per Generation')

    # Sélection du meilleur individu
    best_individual = min(population, key=lambda ind: fitness_func(ind, requests, tasks, users, inputs, outputs,
                                                                   server_processing_capacity))
    data = {
        'generations': generations,
        'best_fitness_per_generation': best_fitness_per_generation,
        'best_count_per_generation': best_count_per_generation,
        'nrequests': len(requests),
        # 'population': population,
        'best_individual': best_individual,
        'best_fitness': fitness_func(best_individual, requests, tasks, users, inputs, outputs,
                                     server_processing_capacity),
        'best_count': count_individual(best_individual, requests, tasks, users, inputs, outputs,
                                       server_processing_capacity),
        'best_groups': form_send_group(best_individual, requests, True)
    }

    return data
