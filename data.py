import numpy as np


class Base:
    def __repr__(self):
        return f"{type(self).__name__}(id:{self.id})"


class Task(Base):
    def __init__(self, task_id, processing_requirement):
        self.id = task_id
        self.processing_requirement = processing_requirement


class User(Base):
    def __init__(self, user_id, computation_capacity, rate):
        self.computation_capacity = computation_capacity
        self.id = user_id
        self.rate = rate


class Request(Base):
    def __init__(self, req_id, user, task, input_index, arrival_time, deadline):
        self.id = req_id
        self.user = user
        self.task = task
        self.input_index = input_index
        self.arrival_time = arrival_time
        self.deadline = deadline


def generata_sytem(ntasks, nusers, ninputs, nrequests, min_task_process,
                   max_task_process, user_computation, min_input, max_input,
                   min_output_time, max_output_time, min_arrival, max_arrival, min_deadline, max_dealine,
                   min_rate, max_rate):
    tasks = [Task(i, np.random.randint(min_task_process, max_task_process)) for i in range(ntasks)]
    users = [User(i, user_computation, np.random.randint(min_rate, max_rate)) for i in range(nusers)]
    inputs = np.random.randint(min_input, max_input, size=ninputs)
    out_times = np.random.uniform(min_output_time, max_output_time, size=(ntasks, ninputs))
    outputs = inputs*out_times
    requests = []
    for i in range(nusers):
        task_id = np.random.randint(0, ntasks)
        input_id = np.random.randint(0, ninputs)
        arrival = np.random.randint(min_arrival, max_arrival)
        deadline = np.random.randint(min_deadline, max_dealine)
        requests.append(Request(i, i, task_id, input_id, arrival, deadline))

    return tasks, users, inputs, outputs, requests
