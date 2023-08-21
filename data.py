import math

import numpy as np

# Parameters for the channel and signal-to-noise ratio
BANDWIDTH = 1e6  # 1 MHz bandwidth
SNR = 10  # Signal-to-noise ratio in dB


class Base:
    def __repr__(self):
        return f"{type(self).__name__}(id:{self.id})"


class Task(Base):
    def __init__(self, task_id, processing_requirement):
        self.id = task_id
        self.processing_requirement = processing_requirement


class User(Base):
    def __init__(self, user_id, computation_capacity, x, y):
        self.computation_capacity = computation_capacity
        self.id = user_id
        self.x = x
        self.y = y
        self.rate = self.shannon_data_rate(math.dist([x, y], [0, 0]), BANDWIDTH, SNR)

    def shannon_data_rate(self, distance, bandwidth, snr):
        return bandwidth * np.log2(1 + snr / distance ** 2)


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
                   min_output_time, max_output_time, min_arrival, max_arrival, min_deadline, max_deadline,
                   mec_radius):
    tasks = [Task(i, np.random.randint(min_task_process, max_task_process)) for i in range(ntasks)]
    # Generate user positions within the MEC's radius
    # https://programming.guide/random-point-within-circle.html
    user_positions = [(mec_radius * np.sqrt(np.random.uniform(0, 1)),
                       np.random.uniform(0, 2 * np.pi)) for _ in range(nusers)]

    users = [User(i, user_computation, x, y) for i, (r, theta) in enumerate(user_positions)
             for x, y in [(r * np.cos(theta), r * np.sin(theta))]]
    inputs = np.random.randint(min_input, max_input, size=ninputs)
    out_times = np.random.uniform(min_output_time, max_output_time, size=(ntasks, ninputs))
    outputs = inputs * out_times
    requests = []
    ri = 0
    for time in range(min_arrival, max_arrival):
        num_requests = np.random.randint(0, nrequests + 1)  # Generate a random number of requests per interval
        userslist = list(range(nusers))
        for _ in range(num_requests):
            user_id = np.random.choice(userslist)
            userslist.remove(user_id)
            task_id = np.random.randint(0, ntasks)
            input_id = np.random.randint(0, ninputs)
            arrival = time
            deadline = np.random.randint(min_deadline, max_deadline)
            requests.append(Request(ri, user_id, task_id, input_id, arrival, deadline))
            ri += 1

    return tasks, users, inputs, outputs, requests


def mock_system():
    users = [User(1, 15, 5, 5), User(2, 15, 0, 5), User(3, 15, 5, 0), User(4, 15, -5, -5), User(5,15, -5, 0)]
    tasks = [Task(1, 10), Task(2, 15), Task(3, 12), Task(4, 18), Task(5, 20)]
    inputs =[500, 1000, 800, 400, 600]
    outputs =[]
    for task in tasks:
        outputs.append([size*task.processing_requirement/10 for size in inputs])

    requests = [Request(0, 1,1,1,0,3),
                Request(1, 2, 1, 2, 0, 3),
                Request(2, 3, 2, 2, 1, 3),
                Request(3, 4, 4, 3, 1, 3),
                Request(4, 0, 3, 2, 1, 3)
                ]

    return tasks, users, inputs, outputs, requests
