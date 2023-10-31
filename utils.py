import matplotlib.pyplot as plt
from collections import defaultdict
import random



LOCAL_LOCATION = 1
SERVER_LOCATION = 2
SKIP_ACTION_VALUE = None  # (-1,-1)
NON_SCHEDULE_PENALITY = 10

def plot(x, y, fmt='o-', xlabel=None, ylabel=None, title=None, grid=False):
    """

    :param x:
    :param y:
    :param fmt:
    :param xlabel:
    :param ylabel:
    :param title:
    :param grid:
    :return:
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, fmt)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(grid)
    plt.show()

    return plt




def is_valid_slot_scheduled(group):
    if len(group):
        group_location, send_time = group[0][1]  # action
        group_input = group[0][0].input_index
        group_task = group[0][0].task

        for req, action in group:
            if group_location == LOCAL_LOCATION:
                if req.input_index == group_input and group_location == LOCAL_LOCATION:
                    continue
                else:
                    return False
            else:
                if req.task == group_task:
                    continue
                else:
                    return False
    return True


def is_valid_individual(individual, tasks, users, requests):
    send_times = defaultdict(list)
    for i, (action, req) in enumerate(zip(individual, requests)):
        if action is SKIP_ACTION_VALUE:
            continue

        location, send_time = action
        send_times[send_time].append((location, req.input_index, req.task))
    for send_time, group in send_times.items():
        group_location = None
        group_input = 0
        group_task = 0
        if len(group):
            group_location, group_input, group_task = group[0]
        for element in group:
            if group_location == LOCAL_LOCATION:
                if element[1] == group_input and group_location == LOCAL_LOCATION:
                    continue
                else:
                    return False
            else:
                if element[2] == group_task:
                    continue
                else:
                    return False

    return True


def form_send_group(individual, requests, with_skip=False):
    send_times = defaultdict(list)
    for i, (action, req) in enumerate(zip(individual, requests)):

        position = i
        if action is SKIP_ACTION_VALUE:
            if with_skip:
                send_times[-1].append((-1, req.input_index, req.task, position))

        else:
            location, send_time = action
            send_times[send_time].append((location, req.input_index, req.task, position))
    return send_times

def reverse_send_group(send_times, num_requests, with_skip=False):
    individual = [SKIP_ACTION_VALUE] * num_requests
    for send_time, group in send_times.items():
        if send_time == -1 and with_skip:
            for item in group:
                location, input_index, task, position = item
                individual[position] = SKIP_ACTION_VALUE
        else:
            for item in group:
                location, input_index, task, position = item
                # print(individual)
                # print("len",num_requests, "position", position )
                individual[position] = (location, send_time)
    return individual
def group_has_conflict(group, requests):
    if len(group) > 1:
        for i in range(len(group)):
            l1, i1, c1, pos1 = group[i]

            for j in range(i + 1, len(group)):
                l2, i2, c2, pos2 = group[j]

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
                    return True

    return False
def has_conflict(individual, requests):
    # Goup = [(location, input, task, position dans individual)]
    groups = form_send_group(individual, requests, with_skip=False)
    for time, group in groups.items():
        has = group_has_conflict(group, requests)
        if has:
            return True

    return False


def individual_conflict_corrector(individual, requests):
    # Goup = [(location, input, task, position dans individual)]
    groups = form_send_group(individual, requests, with_skip=False)
    for time, group in groups.items():
        if len(group) <= 1:
            # pas besoin de change
            continue
        for i in range(len(group) - 1, -1, -1):
            l1, i1, c1, pos1 = group[i]
            valid = True

            for j in range(len(group) - 2, -1, -1):
                l2, i2, c2, pos2 = group[j]

                if c1 == c2 and i1 == i2 and l1 != l2:
                    if random.random() < 0.5:
                        l1 = l2  # Ajuste l'emplacement pour égaliser
                    else:
                        l2 = l1
                elif c1 != c2 and i1 == i2 and l1 == l2:
                    if l1 != LOCAL_LOCATION:
                        l1 = l2 = LOCAL_LOCATION  # Ajuste à LOCAL
                    else:
                        valid = False  # Marque la première réponse comme supprimée
                        break
                elif c1 == c2 and i1 == i2 and l1 == l2:
                    continue
                else:
                    valid = False
                    break

            if valid:
                individual[pos1] = (l1, time)

            else:
                individual[pos1] = SKIP_ACTION_VALUE
                del group[i]

    return individual


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
        input_fixed = False
        lacation = False
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
                if group_input != -1 and group_location != -1:
                    pass
                if group_input != -1 and group_location == -1:

                    individual[position] = (LOCAL_LOCATION, send_time)
                elif task_fixed and group_input == -1:
                    # Ici l faut decide de  ne pas execute certaines requetes
                    skip_id = np.random.randint(len(group))
                    element_to_skip = group[skip_id]
                    _, _, _, position = element_to_skip
                    individual[position] = SKIP_ACTION_VALUE
                    group.remove(element_to_skip)
                if group_input != -1:
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


def multicast_unicast(individual, requests):
    groups = form_send_group(individual, requests, with_skip=False)
    multi = 0
    uni = 0
    for send_time, group in groups.items():
        if len(group ) > 1:
            multi+=1
        else :
            uni+=1
    return  multi, uni


def tikzplotlib_fix_ncols(obj):
    """
    workaround for matplotlib 3.6 renamed legend's _ncol to _ncols, which breaks tikzplotlib
    """
    if hasattr(obj, "_ncols"):
        obj._ncol = obj._ncols
    for child in obj.get_children():
        tikzplotlib_fix_ncols(child)


