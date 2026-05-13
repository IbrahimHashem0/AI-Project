from environment import create_env, build_distance_matrix
from genetic_algorithm import GeneticAlgorithm
from visualization import visualize_solution


def calculate_greedy_solution(distance_matrix):
    """
    Calculates a greedy (nearest neighbor) solution for comparison.
    """
    greedy_order = []
    remaining = set(range(1, 6))  # Victims 1-5
    current = 0  # Start from robot
    total_distance = 0

    while remaining:
        # Find nearest unvisited victim
        nearest = min(remaining, key=lambda v: distance_matrix[current][v])
        greedy_order.append(nearest)
        total_distance += distance_matrix[current][nearest]
        current = nearest
        remaining.remove(nearest)

    return greedy_order, total_distance


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print(" RESCUE MISSION ORDER - TSP VARIANT ")
    print(" Comparative Study: BFS + Genetic Algorithm ")
    print("=" * 60)

    # Step 1: Create environment
    print("\nStep 1: Creating complex environment...")
    grid, robot_pos, victims = create_env()
    print(f"✓ Environment created: {grid.shape[0]}x{grid.shape[1]} grid")
    print(f"✓ Robot position: {robot_pos}")
    print(f"✓ Victim positions: {victims}")

    # Step 2: Build distance matrix using BFS
    print("\nStep 2: Computing distances using BFS...")
    distance_matrix, path_dict = build_distance_matrix(grid, robot_pos, victims)
    print("\nDistance Matrix (Robot=0, V1-V5=1-5):")
    print(distance_matrix)

    # Step 3: Calculate greedy baseline
    print("\nStep 3: Computing greedy baseline solution...")
    greedy_order, greedy_distance = calculate_greedy_solution(distance_matrix)
    print(f"Greedy Solution Order: {greedy_order}")
    print(f"Greedy Total Distance: {greedy_distance:.2f} steps")

    # Step 4: Run Genetic Algorithm
    print("\nStep 4: Running Genetic Algorithm optimization...")
    ga = GeneticAlgorithm(
        distance_matrix=distance_matrix,
        population_size=50,
        mutation_rate=0.15,
        crossover_rate=0.8,
        elite_size=5
    )

    best_order, best_distance = ga.evolve(generations=150, verbose=True)

    # Step 5: Compare results
    print("\n" + "=" * 60)
    print("FINAL COMPARISON: GREEDY vs GENETIC ALGORITHM")
    print("=" * 60)
    print(f"Greedy Solution:")
    print(f"  - Visit Order: {greedy_order}")
    print(f"  - Total Distance: {greedy_distance:.2f} steps")
    print(f"\nGenetic Algorithm Solution:")
    print(f"  - Visit Order: {best_order}")
    print(f"  - Total Distance: {best_distance:.2f} steps")
    print(f"\nImprovement: {greedy_distance - best_distance:.2f} steps")
    print(f"Percentage Better: {((greedy_distance - best_distance) / greedy_distance * 100):.2f}%")
    print("=" * 60 + "\n")

    # Step 6: Visualize solution
    print("Step 5: Generating visualization...")
    visualize_solution(grid, robot_pos, victims, best_order, path_dict, distance_matrix, ga)

    print("\n✓ PROJECT COMPLETED SUCCESSFULLY!\n")
