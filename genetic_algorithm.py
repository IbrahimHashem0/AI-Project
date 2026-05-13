import numpy as np
import random
import time

class GeneticAlgorithm:
    """
    Genetic Algorithm for optimizing the rescue mission order.
    Chromosome Representation: A permutation of [1, 2, 3, 4, 5]
    representing the order in which victims should be rescued.
    Example: [3, 1, 5, 2, 4] means visit V3 first, then V1, then V5, V2, V4.
    """

    def __init__(self, distance_matrix, population_size=100,
                 mutation_rate=0.15, crossover_rate=0.8, elite_size=10):
        """
        Initialize the Genetic Algorithm.
        """
        self.distance_matrix = distance_matrix
        self.population_size = population_size # responsible for Search Space Coverage
        self.mutation_rate = mutation_rate # responsible for Genetic Diversity
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size

        # Statistics tracking
        self.best_fitness_history = []
        self.avg_fitness_history = []

    def calculate_fitness(self, chromosome):
        """
        CRITICAL: Fitness Function Calculation
        Fitness = Total distance traveled following the rescue order in chromosome
        """
        total_distance = 0.0

        # Start from Robot (index 0) to first victim
        current_location = 0  # Robot is at index 0

        for victim_id in chromosome:
            # victim_id ranges from 1-5, which correspond to indices 1-5 in distance_matrix
            next_location = victim_id

            # Add distance from current to next location
            total_distance += self.distance_matrix[current_location][next_location]

            # Update current location
            current_location = next_location

        return total_distance

    def create_initial_population(self):
        """
        Creates the initial population of random chromosomes.
        """
        population = []
        base_chromosome = [1, 2, 3, 4, 5]

        for _ in range(self.population_size):
            chromosome = base_chromosome.copy()
            random.shuffle(chromosome)
            population.append(chromosome)

        return population

    def selection(self, population, fitnesses):
        """
        Tournament Selection: randomly select K individuals and pick the best.
        """
        tournament_size = 3
        tournament_indices = random.sample(range(len(population)), tournament_size)

        # Find the best chromosome in the tournament (lowest fitness)
        best_idx = min(tournament_indices, key=lambda i: fitnesses[i])

        return population[best_idx].copy()

    def ordered_crossover(self, parent1, parent2):
        """
        Ordered Crossover (OX) - preserves relative order from parents.
        """
        size = len(parent1)

        # Select two random crossover points
        point1, point2 = sorted(random.sample(range(size), 2))

        # Create child1
        child1 = [None] * size
        child1[point1:point2] = parent1[point1:point2]

        # Fill remaining positions from parent2
        pointer = point2
        for gene in parent2[point2:] + parent2[:point2]:
            if gene not in child1:
                if pointer >= size:
                    pointer = 0
                child1[pointer] = gene
                pointer += 1

        # Create child2 (swap parents)
        child2 = [None] * size
        child2[point1:point2] = parent2[point1:point2]

        pointer = point2
        for gene in parent1[point2:] + parent1[:point2]:
            if gene not in child2:
                if pointer >= size:
                    pointer = 0
                child2[pointer] = gene
                pointer += 1

        return child1, child2

    # Swap Mutation: randomly swap two positions in the chromosome.
    def swap_mutation(self, chromosome):
        mutated = chromosome.copy()

        if random.random() < self.mutation_rate:
            # Select two random positions
            idx1, idx2 = random.sample(range(len(mutated)), 2)
            # Swap them
            mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]

        return mutated

    def evolve(self, generations=500, verbose=True):
        """
        Main evolution loop for the Genetic Algorithm.
        """
        print("\n" + "=" * 60)
        print("STARTING GENETIC ALGORITHM OPTIMIZATION")
        print("=" * 60)
        print(f"Population Size: {self.population_size}")
        print(f"Mutation Rate: {self.mutation_rate}")
        print(f"Crossover Rate: {self.crossover_rate}")
        print(f"Elite Size: {self.elite_size}")
        print(f"Generations: {generations}")
        print("=" * 60 + "\n")

        start_time = time.time()

        # Create initial population
        population = self.create_initial_population()

        best_overall_chromosome = None
        best_overall_fitness = float('inf')

        for generation in range(generations):
            # Calculate fitness for all chromosomes
            fitnesses = [self.calculate_fitness(chrom) for chrom in population]

            # Find best in this generation
            best_idx = fitnesses.index(min(fitnesses))
            best_fitness = fitnesses[best_idx]
            best_chromosome = population[best_idx]

            # Track statistics
            self.best_fitness_history.append(best_fitness)
            self.avg_fitness_history.append(np.mean(fitnesses))

            # Update overall best
            if best_fitness < best_overall_fitness:
                best_overall_fitness = best_fitness
                best_overall_chromosome = best_chromosome.copy()

            # Print progress
            if verbose and (generation % 10 == 0 or generation == generations - 1):
                worst_idx = fitnesses.index(max(fitnesses))
                worst_chromosome = population[worst_idx]

                print(f"Generation {generation:3d} | Best Fitness: {best_fitness:.2f} | "
                      f"Avg Fitness: {np.mean(fitnesses):.2f} | "
                      f"Best Order: {best_chromosome}")

                print(
                    f"   --> $DEBUG$: Mutation trying other paths: {worst_chromosome} (Fitness: {fitnesses[worst_idx]})")

            # Elitism: preserve top chromosomes
            sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i])
            elite = [population[i].copy() for i in sorted_indices[:self.elite_size]]

            # Create new population
            new_population = elite.copy()

            while len(new_population) < self.population_size:
                # Selection
                parent1 = self.selection(population, fitnesses)
                parent2 = self.selection(population, fitnesses)

                # Crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = self.ordered_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation
                child1 = self.swap_mutation(child1)
                child2 = self.swap_mutation(child2)

                new_population.extend([child1, child2])

            # Trim to exact population size
            population = new_population[:self.population_size]

        elapsed_time = time.time() - start_time

        print("\n" + "=" * 60)
        print("GENETIC ALGORITHM COMPLETED")
        print("=" * 60)
        print(f"Time Elapsed: {elapsed_time:.2f} seconds")
        print(f"Best Solution Found: {best_overall_chromosome}")
        print(f"Total Distance: {best_overall_fitness:.2f} steps")
        print("=" * 60 + "\n")

        return best_overall_chromosome, best_overall_fitness
