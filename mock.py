from data import mock_system
from genetic import genetic_algorithm, fitness_soft, evaluate_individual




SERVER_COMPUTATION_CAPACITY = 50
# Utilisation de l'algorithme génétique
POPULATION_SIZE = 100
GENERATIONS = 100
MUTATION_RATE = 0.1
PROBABILITY_SKIP = 0.05

tasks, users, inputs, outputs, requests = mock_system()

result = genetic_algorithm(tasks, users, requests, inputs, outputs, SERVER_COMPUTATION_CAPACITY, POPULATION_SIZE,
                           GENERATIONS,
                           MUTATION_RATE, PROBABILITY_SKIP, draw=True, fitness_func=evaluate_individual)
print(requests)
[print(key, val) for key, val in result.items()]