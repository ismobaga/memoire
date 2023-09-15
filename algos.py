from collections import defaultdict
import numpy as np
from genetic import  LOCAL_LOCATION, SERVER_LOCATION, evaluate_individual, \
    count_individual
from utils import reverse_send_group, has_conflict


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

                individual = reverse_send_group(groups, len(requests), True)
                if has_conflict(individual, requests):
                    # Les -1 ne peuvent pas avoir de conflict donc ignore
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
            individual = reverse_send_group(groups, len(requests), True)
            score = count_individual(individual, requests, tasks, users, inputs, outputs, server_processing_capacity)
            loc_scores[location] = score

        # Compare les deux version et choisi le meilleur
        # print(loc_scores)
        location = max(loc_scores, key=loc_scores.get)
        # print("max loc", location, loc_scores[location])
        final_groups = loc_groups[location]
        # print("score", score, "position", position, "restant", len(urequests))
        # print("final", final_groups)
        # print("local", loc_groups[LOCAL_LOCATION])
        # print("serve", loc_groups[SERVER_LOCATION])
        position += 1

    individual = reverse_send_group(final_groups, len(requests), True)
    # print("fin heuristic")
    return individual, count_individual(individual, requests, tasks, users, inputs, outputs, server_processing_capacity)

