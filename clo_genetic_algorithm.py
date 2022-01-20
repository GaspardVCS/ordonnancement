import numpy as np
from utils import generate_jobs_sample, generate_machines_sample
from copy import deepcopy

seed = 18
# np.random.seed(seed)

class CloGeneticAlgorithm:
    def __init__(self, jobs, machines) -> None:
        self.jobs = jobs
        self.machines = machines
        self.prop_to_mutate = 0.3
        self.population_size = 100
        self.verbose = False
        self.population = self.random_population(self.population_size)

    def random_allocation(self):
        """
        Generate a random allocation, ie a list of the different tasks
        The order in the list is the order of starting times of the tasks
        Parameters:
            jobs (list): list of the different jobs. 
        Returns:
            allocation (list): list of tasks sorted by starting time
        """
        jobs_copy = deepcopy(self.jobs)
        tasks_order = np.random.permutation(list(range(len(self.jobs))) * 3)
        allocation = []
        for job_index in tasks_order:
            allocation.append(jobs_copy[job_index].pop(0))
        return allocation   

    def mutation(self, allocation):
        """
        Mutate an allocation by randomly permuting two neighborings tasks from different job
        The implementation is recursive, but one can easily change that.
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
            return self.mutation(allocation)
    
    @staticmethod
    def crossover1(allocation1, allocation2):
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
    
    @staticmethod
    def crossover2(allocation1, allocation2):
        """
        For each children, we keep the second half of each allocation
        and add the remaining tasks in the order of the other allocation
        Parameters:
            allocation1 (list): first allocation
            allocation2 (list): second allocation
        Returns:
            cross_allocation1 (list): first child allocation
            cross_allocation2 (list): second child allocation
        """
        n = len(allocation1) // 2
        cross_allocation1 = allocation1[n:]
        cross_allocation2 = allocation2[n:]
        for task in allocation2[::-1]:
            if task not in cross_allocation1:
                cross_allocation1 = [task] + cross_allocation1
        for task in allocation1[::-1]:
            if task not in cross_allocation2:
                cross_allocation2 = [task] + cross_allocation2
        return cross_allocation1, cross_allocation2


    def finished_times(self, allocation):
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
            start_breakdown, end_breakdown = self.machines[machine_index]
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
        

    def fitness_function(self, allocation):
        """
        Calculate the cost of an allocation
        Parameters:
            allocation (list)
            jobs (list)
            machines (list)
        Returns:
            cost (int)
        """
        finishing_times = self.finished_times(allocation)
        cost = 0
        for i, job in enumerate(self.jobs):
            job_finish_time = finishing_times[i][-1]
            if job_finish_time > job[-2]:
                cost += job[-1]
        return cost

    def random_population(self, population_size=20):
        """
        Generate a random list of feasible allocations
        Parameters:
            population_size (int): number of allocation to generate
        Returns:
            population (list): list of allocations     
        """
        population = [self.random_allocation() for _ in range(population_size)]
        return population

    def next_generation(self):
        """
        Generate the next generation. Update the self.population attribute
        Keep the 50% best allocations from the old generation.
        Then create the remaining 50% with crossovers between two allocations 
        of the best 50% of the old generation.
        Randomly mutate chromosomes in the resulting population.
        Parameters: None
        Returns: None
        """
        old_generation = sorted(self.population, key=lambda x: self.fitness_function(x))
        n = len(old_generation) // 2
        new_generation = old_generation[:n]
        while len(new_generation) < len(old_generation):
            p = np.random.uniform(0, 1)
            i, j = np.random.randint(n, size=2)
            child1, child2 = self.crossover1(new_generation[i], new_generation[j]) if p <= 0.5 \
                                     else self.crossover2(new_generation[i], new_generation[j])
            if child1 not in new_generation:
                new_generation.append(child1)
            if child2 not in new_generation:
                new_generation.append(child2)

        for i, a in enumerate(new_generation):
            p = np.random.uniform(0, 1)
            if p <= self.prop_to_mutate:
                new_generation[i] = self.mutation(a)
        self.population = new_generation
        mean_cost = np.mean(list(map(lambda x: self.fitness_function(x), new_generation)))
        if self.verbose:
            print(f'{self.fitness_function(self.best_allocation())}')
            print(f'number of late jobs: {self.number_late_jobs(self.best_allocation())}')
    
    def number_late_jobs(self, allocation):
        finishing_times = self.finished_times(allocation)
        late_jobs = 0
        for i, job in enumerate(self.jobs):
            job_finish_time = finishing_times[i][-1]
            if job_finish_time > job[-2]:
                late_jobs += 1
        return late_jobs

    def best_allocation(self):
        """
        Get the best allocation in the current population
        Returns:
            best allocation
        """
        return min(self.population, key=lambda x: self.fitness_function(x))