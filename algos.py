
def schedule_tasks(tasks, users, inputs, outputs, requests):
    schedule = {}  # Dictionnaire pour stocker la planification des tâches
    # Trier les requêtes par ordre d'arrivée
    requests.sort(key=lambda req: req.arrival_time)

    for req in requests:
        task = tasks[req.task]
        user = users[req.user]
        input_size = inputs[req.input_index]
        output_size = outputs[req.task][req.input_index]

#         Choisir le dernier qui permet d'envoyer la reponse  et ajoute le plus posible de requete




def schedule_tasks(tasks, users, requests):
    schedule = {}  # Dictionnaire pour stocker la planification des tâches

    # Trier les requêtes par ordre d'arrivée
    requests.sort(key=lambda req: req.arrival_time)

    for req in requests:
        task = tasks[req.task]
        user = users[req.user]

        # Vérifier si le temps d'exécution de la tâche dépasse le délai
        execution_time = task.processing_requirement / user.computation_capacity
        if execution_time > req.deadline - req.arrival_time:
            continue  # Ignorer cette requête si le délai est dépassé

        # Planifier la tâche pour exécution
        if user.id not in schedule:
            schedule[user.id] = []
        schedule[user.id].append(req)

    return schedule


# Utilisation du code généré
scheduled_tasks = schedule_tasks(tasks, users, requests)

# Afficher la planification des tâches pour chaque utilisateur
for user_id, user_schedule in scheduled_tasks.items():
    print(f"User {user_id}:")
    for req in user_schedule:
        print(
            f"  Request {req.id}: Task {req.task}, Input {req.input_index}, Arrival {req.arrival_time}, Deadline {req.deadline}")
