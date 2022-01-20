from xml.dom import VALIDATION_ERR
from tqdm import tqdm
from utils import generate_jobs_sample, generate_machines_sample
from clo_genetic_algorithm import CloGeneticAlgorithm

# Set verbose to true if you want to see the prints
VERBOSE = True

# Number of generations 
n_generation = 500


if __name__ == '__main__':
    # generate random jobs and random machines
    jobs = generate_jobs_sample(n_jobs=6)
    machines = generate_machines_sample()

    # Print the jobs and machines
    if VERBOSE:
        for i, job in enumerate(jobs):
            print(f'Job {i}: {job}')
        print()
        for i, machine in enumerate(machines):
            print(f'machine {i}: {machine}')
        print()

    # Instantiate the CloGeneticAlgorithm object
    cga = CloGeneticAlgorithm(jobs=jobs, machines=machines)

    # print the initial score and initial number of late jobs
    if VERBOSE:
        best_allocation = cga.best_allocation()
        late_jobs = cga.number_late_jobs(best_allocation)
        cost = cga.fitness_function(best_allocation)
        print(f'Initial number of late jobs: {late_jobs}')
        print(f'The initial allocation has a cost of: {cost}')
        print()

    # Develop the population for n_generation
    for _ in tqdm(range(n_generation)):
        cga.next_generation()

    # print the initial score and initial number of late jobs
    if VERBOSE:
        best_allocation = cga.best_allocation()
        late_jobs = cga.number_late_jobs(best_allocation)
        cost = cga.fitness_function(best_allocation)
        print(f'Final number of late jobs: {late_jobs}')
        print(f'The final allocation has a cost of: {cost}')