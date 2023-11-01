from collections import defaultdict
import numpy as np
from genetic import  LOCAL_LOCATION, SERVER_LOCATION, evaluate_individual, \
    count_individual
from utils import reverse_send_group, has_conflict, group_has_conflict


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


def glouton(tasks, users, requests, inputs, outputs, server_processing_capacity):
    urequests = requests.copy()
    final_groups = defaultdict(list)
    position = 0
    while len(urequests)> 0:
        base_req = urequests.pop(0)
def heuristic(tasks, users, requests, inputs, outputs, server_processing_capacity):
    urequests = requests.copy()
    final_groups = defaultdict(list)
    position = 0
    while len(urequests):
        base_req = urequests.pop(0)
        toremove = {LOCAL_LOCATION:[], SERVER_LOCATION:[]} # la groupe g'envoie a suprime

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

                individual = reverse_send_group(groups, len(requests), True)
                if has_conflict(individual, requests):
                    # Les -1 ne peuvent pas avoir de conflict donc ignore
                    groups.pop(sendtime)
                    groups = cgroups
                else:
                    time_for_group = sendtime
                    break
            if time_for_group!=-1:
                for sub_position, req in enumerate(urequests):
                    individual = reverse_send_group(groups, len(requests), True)
                    prev_score = count_individual(individual, requests, tasks, users, inputs, outputs,
                                             server_processing_capacity)
                    groups[time_for_group].append((location, req.input_index, req.task, position + sub_position))
                    if len(requests) <= (position + sub_position):
                        print(position, sub_position)
                    individual = reverse_send_group(groups, len(requests))
                    if has_conflict(individual, requests):
                        groups[time_for_group].pop()
                    else:
                        individual = reverse_send_group(groups, len(requests), True)
                        score = count_individual(individual, requests, tasks, users, inputs, outputs,
                                                 server_processing_capacity)
                        if prev_score > score:
                            groups[time_for_group].pop()

            toremove[location] = groups[time_for_group]
            loc_groups[location] = groups
            individual = reverse_send_group(groups, len(requests), True)
            score = count_individual(individual, requests, tasks, users, inputs, outputs, server_processing_capacity)
            loc_scores[location] = score

        # Compare les deux version et choisi le meilleur
        # print(loc_scores)
        location = max(loc_scores, key=loc_scores.get)
        # print("location", location)
        # print("max loc", location, loc_scores[location])
        final_groups = loc_groups[location]

        # print("to remove", toremove[location])
        for l, i_id, t_id, pos in toremove[location]:
            req = requests[pos]
            # urequests.remove((pos-position, r_id))
            urequests = list(filter(lambda x: not(x.id == req.id and req.task == x.task), urequests))

        # print("score", loc_scores, "position", position, "restant", len(urequests),"/", len(requests))
        # print("final", final_groups)
        # print("local", loc_groups[LOCAL_LOCATION])
        # print("serve", loc_groups[SERVER_LOCATION])
        position += 1

    individual = reverse_send_group(final_groups, len(requests), True)
    print("individu", individual)
    print("group", final_groups)
    # print("fin heuristic")
    return individual, count_individual(individual, requests, tasks, users, inputs, outputs, server_processing_capacity)



def online(tasks, users, requests, inputs, outputs, server_processing_capacity):
    current_time = 0
    request_queue = []
    final_groups = defaultdict(list)
    do = True

    while do or request_queue:
        # un do while
        do = False
        # Trie les requêtes en fonction de leur deadline
        request_queue.sort(key=lambda request: request.arrival_time + request.deadline)

        # Sélectionne la première requête dans la file
        next_request = request_queue[0]
        if next_request.arrival_time <= current_time:
            # on le planifi pour current time
            # on va determile le quel des deux mode d'envoi peux accommoder le plusde requetes
            sendtime = current_time
            groups = {LOCAL_LOCATION: [], SERVER_LOCATION:[]}
            for location in [LOCAL_LOCATION, SERVER_LOCATION]:
                group = []
                group.append((location, next_request.input_index, next_request.task, next_request.id))
                for request in request_queue:
                    group.append((location, request.input_index, request.task, request.id))
                    has = group_has_conflict(group, requests)
                    if has:
                        # si on ne peut pas ajoute au groupe d'envoi sans conflit
                        group.pop()

                groups[location] = group
            # Comparer les deux groups choisir le meuilleur

        # Enleve les requete scheduler de la fille
        # Increment le temps pour passe a l'instant suivant

        # Evaler le tout
