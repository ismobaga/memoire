import numpy as np

# Paramètres généraux
num_users = 3
num_tasks = 5
num_inputs = 10
num_intervals = 10

# Génération aléatoire des capacités de calcul pour chaque utilisateur
user_capabilities = np.random.randint(5, 20, size=num_users)

# Génération aléatoire des temps d'exécution pour chaque tâche au niveau de l'utilisateur et du MEC
execution_times_local = np.random.randint(2, 10, size=num_tasks)
execution_times_mec = np.random.randint(5, 20, size=num_tasks)

# Génération aléatoire des échéances pour chaque requête
deadlines = np.random.randint(5, 20, size=(num_users, num_inputs))

# Génération aléatoire des temps d'arrivée pour chaque requête
arrival_times = np.random.randint(1, num_intervals, size=(num_users, num_inputs))

# Fonction de calcul du temps total pour une tâche donnée
def calculate_total_time(execute_local, user, task, input_index):
    if execute_local:
        execution_time = execution_times_local[task] / user_capabilities[user]
    else:
        execution_time = execution_times_mec[task] / mec_capability
    return execution_time

# Algorithme d'ordonnancement
def schedule_requests(requests):
    scheduled_requests = []

    for user in range(num_users):
        for input_index in range(num_inputs):
            task = np.random.randint(num_tasks)  # Choix aléatoire de la tâche
            execute_local = True  # Supposons que nous essayons d'exécuter localement par défaut
            total_time_local = calculate_total_time(True, user, task, input_index)
            total_time_mec = calculate_total_time(False, user, task, input_index)

            if total_time_local > deadlines[user, input_index]:
                execute_local = False

            if not execute_local and total_time_mec <= deadlines[user, input_index]:
                scheduled_requests.append((user, task, input_index, execute_local))

    return scheduled_requests

# Capacité de calcul du MEC
mec_capability = np.random.randint(10, 30)

print('Exemples de requêtes générées:')
for user in range(num_users):
    for input_index in range(num_inputs):
        task = np.random.randint(num_tasks)
        arrival_time = arrival_times[user, input_index]
        deadline = deadlines[user, input_index]
        print(f'Utilisateur {user}, Tâche {task}, Entrée {input_index}, Temps d\'arrivée {arrival_time}, Échéance {deadline}')

print('Capacité de calcul du MEC :', mec_capability)

# Ordonnancement des requêtes
requests = [(user, task, input_index, execute_local) for user, task, input_index, execute_local in schedule_requests(requests)]

print('Requêtes ordonnancées:')
for user, task, input_index, execute_local in requests:
    execution_location = 'local' if execute_local else 'MEC'
    print(f'Utilisateur {user}, Tâche {task}, Entrée {input_index}, Exécution {execution_location}')
