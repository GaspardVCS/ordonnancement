import numpy as np

def generate_jobs_sample(n_jobs=5):
    """
    Generate a n random jobs.
    A job is a list of 3 * (machine index, time of the task, job_index) and two integers, one 
    that represents the time after which a job is considered late and one that represents the cost of
    not finishing the job.
    Paremeters:
        n_jobs (int): number of jobs to generate
    Returns:
        jobs (list): list of jobs generated 
    """
    jobs = []
    for i in range(n_jobs):
        machines_assignement = np.random.permutation([1, 2, 3]) # random machine assignement order
        task_times = np.random.randint(1, 6, size=3) # time taken by each task 
        due_date = np.sum(task_times) + int(np.random.randint(15, size=1)) # if the job finishes after that date, it is considered late
        cost = np.random.randint(10)
        job = list(zip(machines_assignement, task_times, [i] * 3)) + [due_date, cost]
        jobs.append(job)
    return jobs

def generate_machines_sample():
    """
    Generates 3 random machines.
    Each machine is described by a tuple composed of the starting time and the 
    ending time of the breakdown on the machine
    Returns:
        machines (list): list of machines generated
    """
    machines = []
    for _ in range(3):
        start_time = np.random.randint(2, 10) # starting time of the breakdown
        end_time = start_time + np.random.randint(1, 5) # end of the breakdown on the machine
        machines.append((start_time, end_time))
    return machines 