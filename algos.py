from collections import defaultdict
import numpy as np
from genetic import reverse_send_group, LOCAL_LOCATION, SERVER_LOCATION, has_conflict, evaluate_individual, \
    count_individual

# Probleme to fix : le nombre de taches execute diminus  apres des iterations

def can_be_add(group, value):
    if len(group) >= 1:
        l1, i1, c1, pos1 = group[0]

        l2, i2, c2, pos2 = value

        conditions = [
            (c1 == c2 and i1 == i2 and l1 != l2),
            (c1 == c2 and i1 != i2 and l1 != l2),
            (c1 != c2 and i1 != i2 and l1 != l2),
            (c1 == c2 and i1 != i2 and l1 == l2),
            (c1 != c2 and i1 != i2 and l1 == l2 and l1 != LOCAL_LOCATION),
            (c1 != c2 and i1 == i2 and l1 == l2 and l1 != LOCAL_LOCATION),
            (c1 != c2 and i1 == i2 and l1 != l2),
            (c1 != c2 and i1 != i2 and l1 == l2)
        ]

        if any(conditions):
            return False

    return True
def heuristic1(tasks, users, requests, inputs, outputs, server_processing_capacity):
    urequests = requests.copy()
    final_groups = defaultdict(list)
    position = 0

    while len(urequests):
        base_req = urequests.pop(0)

        # Choix plus de temps avant la date limite
        location_choices = [LOCAL_LOCATION, SERVER_LOCATION]
        loc_groups = {location: final_groups.copy() for location in location_choices}
        loc_scores = {location: 0 for location in location_choices}

        for location in location_choices:
            group = []
            groups = loc_groups[location]
            group.append((location, base_req.input_index, base_req.task, position))

            # Trouver un bon temps d'envoi
            possible_times = list(range(base_req.arrival_time, base_req.arrival_time + base_req.deadline))
            possible_times = [i for i in possible_times if i not in groups]
            possible_times.append(-1)
            time_for_group = -1
            cgroups = groups.copy()

            for sendtime in possible_times:
                if sendtime in groups and sendtime != -1:
                    continue
                elif sendtime not in groups:
                    groups[sendtime] = group
                elif sendtime == -1:
                    groups[sendtime].extend(group)

                individual = reverse_send_group(groups, len(requests))
                if has_conflict(individual, requests):
                    groups.pop(sendtime)
                    groups = cgroups
                else:
                    time_for_group = sendtime
                    break

            for sub_position, req in enumerate(urequests):
                groups[time_for_group].append((location, req.input_index, req.task, position + sub_position))
                if len(requests) <= (position + sub_position):
                    print(position, sub_position)
                individual = reverse_send_group(groups, len(requests))
                if has_conflict(individual, requests):
                    groups[time_for_group].pop()

            loc_groups[location] = groups
            individual = reverse_send_group(groups, len(requests))
            score = count_individual(individual, requests, tasks, users, inputs, outputs, server_processing_capacity)
            loc_scores[location] = score
            print("score", score)

        # Compare les deux versions et choisi la meilleure
        print(loc_scores)
        location = max(loc_scores, key=loc_scores.get)
        final_groups = loc_groups[location]
        print("max loc", location, loc_scores[location])
        print("final", final_groups)
        print("local", loc_groups[LOCAL_LOCATION])
        print("serve", loc_groups[SERVER_LOCATION])
        position += 1

    individual = reverse_send_group(final_groups, len(requests))
    return individual, count_individual(individual, requests, tasks, users, inputs, outputs, server_processing_capacity)



def heuristic(tasks, users, requests, inputs, outputs, server_processing_capacity):
    urequests = requests.copy()
    final_groups = defaultdict(list)
    position = 0
    while len(urequests):
        base_req = urequests.pop(0)

        # chox plus de temps avant deadline
        # reageder les deux possibilie
        location = np.random.choice([LOCAL_LOCATION, SERVER_LOCATION])
        loc_groups = {LOCAL_LOCATION: final_groups.copy(), SERVER_LOCATION: final_groups.copy()}
        loc_scores = {LOCAL_LOCATION: 0, SERVER_LOCATION: 0}
        for location in [LOCAL_LOCATION, SERVER_LOCATION]:
            group = []
            groups = loc_groups[location]
            group.append((location, base_req.input_index, base_req.task, position))

            # Trouver un bon temps d'envoie
            # group = [base_req]
            possible_times = list(range(base_req.arrival_time, base_req.arrival_time + base_req.deadline))
            possible_times = [i for i in possible_times if i not in groups]
            # Ajoute la possibilite de ne pas du tout execute la tache
            possible_times.append(-1)
            time_for_group = -1
            cgroups = groups.copy()
            for sendtime in possible_times:
                # il faut verifier que le temps d'execution n'est pas utilise, sauf pour -1
                if sendtime in groups and sendtime != -1:
                    continue
                elif sendtime not in groups :
                    groups[sendtime] = group
                elif sendtime ==-1:
                    # On ajoute a la liste des non execute
                    groups[sendtime].extend(group)

                individual = reverse_send_group(groups, len(requests))
                if has_conflict(individual, requests):
                    # Les -1 ne peuvent pas avoir de conflict donc ignore
                    groups.pop(sendtime)
                    groups = cgroups
                else:
                    time_for_group = sendtime
                    break

            sub_position = 0
            while sub_position < len(urequests):
                groups[time_for_group].append((location, req.input_index, req.task, position + sub_position))
                if len(requests) <= (position + sub_position):
                    print(position, sub_position)
                individual = reverse_send_group(groups, len(requests))
                if has_conflict(individual, requests):
                    groups[time_for_group].pop()
                    sub_position +=1
                else:
                    urequests.pop(sub_position)
           

            loc_groups[location] = groups
            individual = reverse_send_group(groups, len(requests))
            score = count_individual(individual, requests, tasks, users, inputs, outputs, server_processing_capacity)
            loc_scores[location] = score

        # Compare les deux version et choisi le meilleur
        # print(loc_scores)
        location = max(loc_scores, key=loc_scores.get)
        # print("max loc", location, loc_scores[location])
        final_groups = loc_groups[location]
        print("score", score, "position", position, "restant", len(urequests))
        # print("final", final_groups)
        # print("local", loc_groups[LOCAL_LOCATION])
        # print("serve", loc_groups[SERVER_LOCATION])
        position += 1

    individual = reverse_send_group(final_groups, len(requests))
    return individual, count_individual(individual, requests, tasks, users, inputs, outputs, server_processing_capacity)


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
# scheduled_tasks = schedule_tasks(tasks, users, requests)
scheduled_tasks = {}
# Afficher la planification des tâches pour chaque utilisateur
for user_id, user_schedule in scheduled_tasks.items():
    print(f"User {user_id}:")
    for req in user_schedule:
        print(
            f"  Request {req.id}: Task {req.task}, Input {req.input_index}, Arrival {req.arrival_time}, Deadline {req.deadline}")
