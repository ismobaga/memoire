from data import mock_system
from genetic import genetic_algorithm, fitness_soft, evaluate_individual, count_individual
from algos import heuristic




SERVER_COMPUTATION_CAPACITY = 50
# Utilisation de l'algorithme génétique
POPULATION_SIZE = 100
GENERATIONS = 100
MUTATION_RATE = 0.1
PROBABILITY_SKIP = 0.05

def reverse_count(func):
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        return len(requests) - result

    return wrapper

non_exec_count = reverse_count(count_individual)

tasks, users, inputs, outputs, requests = mock_system()

result = genetic_algorithm(tasks, users, requests, inputs, outputs, SERVER_COMPUTATION_CAPACITY, POPULATION_SIZE,
                           GENERATIONS,
                           MUTATION_RATE, PROBABILITY_SKIP, draw=False, fitness_func=non_exec_count)

result2 = heuristic(tasks, users, requests, inputs, outputs, SERVER_COMPUTATION_CAPACITY)

print(requests)
print(result['best_individual'], result['best_count'] )
print(result2)
# [print(key, val) for key, val in result.items()]