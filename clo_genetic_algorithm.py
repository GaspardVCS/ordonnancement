from platform import machine
import numpy as np
from utils import generate_jobs_sample, generate_machines_sample
from copy import deepcopy

# np.random.seed(42)

def random_allocation(jobs):
    """
    Generate a random allocation, ie a list of the different tasks
    The order in the list is the order of starting times of the tasks
    Parameters:
        jobs (list): list of the different jobs. 
    Returns:
        allocation (list): list of tasks sorted by starting time
    """
    jobs_copy = deepcopy(jobs)
    tasks_order = np.random.permutation(list(range(len(jobs))) * 3)
    allocation = []
    for job_index in tasks_order:
        allocation.append(jobs_copy[job_index].pop(0))
    return allocation

def is_allocation_correct(allocation, jobs):
    """
    Checks if an allocation is correct. Ie if all the jobs are made in the 
    correct order
    Parameters:
        allocation (list): an allocation of the tasks sorted in by starting time
        jobs (list): list of jobs
    Returns:
        (bool): true if the allocation is feasible, false otherwise
    """
    tasks_order = sorted(allocation, key=lambda x: x[2])
    for i in range(len(jobs)):
        if tasks_order[3*i:3*i+3] != jobs[i][:-2]:
            return False
    return True   

def mutation(allocation):
    """
    Mutate an allocation by randomly permuting two neighborings tasks from different job
    Parameters:
        allocation (list): allocation list of tasks
    Returns:
        mutated_allocation (list): new allocation
    """
    n = len(allocation)
    index = np.random.randint(n-1)
    mutated_allocation = deepcopy(allocation)
    # We check that we don't have the same job
    if mutated_allocation[index][2] != mutated_allocation[index + 1][2]:
        mutated_allocation[index], mutated_allocation[index + 1] = mutated_allocation[index + 1], mutated_allocation[index]
        return mutated_allocation
    # If we have the same job, we try again with another random index
    else:
        return mutation(allocation)

def crossover(allocation1, allocation2):
    """
    For each children, we keep the first half of each allocation
    and add the remaining tasks in the order of the other allocation
    Parameters:
        allocation1 (list): first allocation
        allocation2 (list): second allocation
    Returns:
        cross_allocation1 (list): first child allocation
        cross_allocation2 (list): second child allocation
    """
    n = len(allocation1) // 2
    cross_allocation1 = allocation1[:n]
    cross_allocation2 = allocation2[:n]
    for task in allocation2:
        if task not in cross_allocation1:
            cross_allocation1.append(task)
    for task in allocation1:
        if task not in cross_allocation2:
            cross_allocation2.append(task)
    return cross_allocation1, cross_allocation2

def finished_times(allocation, machines):
    """
    Array of size (n_jobs, 3)
    Cell i, j get the finishing time of task j for job i
    Parameters:
        allocation (list)
        machines (list)
    Returns:
        finishing_times (array)
    """
    task_finishing_times = np.ones((len(allocation) // 3, 3)) * (-1)
    machine_time = [0, 0, 0]
    for task in allocation:
        machine_index, length, job_index = task
        machine_index -= 1
        start_breakdown, end_breakdown = machines[machine_index]
        machine_current_time = machine_time[machine_index]
        # If the task will intersect with the breakdown of the machine, we start it at the end of the breakdown
        if (machine_current_time + length > start_breakdown) and (machine_current_time <= start_breakdown):
            machine_time[machine_index] = end_breakdown
        # We find which task in the job it is, ie between 0, 1 or 2
        task_order = np.argmin(task_finishing_times[job_index][:])
        # We calculate the starting time of the task
        # It is the max between the current machine time and the end of the previous task on the job, if any
        start_time = max(machine_time[machine_index], task_finishing_times[job_index][task_order - 1]) 
        # The finish time is just the starting time + the length of the task
        finish_time = start_time + length
        # We update the machine current time
        machine_time[machine_index] = finish_time
        # We add the finishing time to the array
        task_finishing_times[job_index][task_order] = finish_time
    return np.array(task_finishing_times)
        

def fitness_function(allocation, jobs, machines):
    """
    Calculate the cost of an allocation
    Parameters:
        allocation (list)
        jobs (list)
        machines (list)
    Returns:
        cost (int)
    """
    finishing_times = finished_times(allocation, machines)
    cost = 0
    for i, job in enumerate(jobs):
        job_finish_time = finishing_times[i][-1]
        if job_finish_time > job[-2]:
            cost += job[-1]
    return cost

if __name__ == '__main__':
    # Generate the data randomly
    jobs = generate_jobs_sample(n_jobs=3)
    machines = generate_machines_sample()

    # Create a random allocation
    allocation = random_allocation(jobs=jobs)

    # calculate its cost
    cost = fitness_function(allocation, jobs, machines)
    print(cost)