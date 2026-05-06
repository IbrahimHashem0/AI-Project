import numpy as np
from collections import deque

# ============================================================================
# PART 1: ENVIRONMENT SETUP AND BFS PATHFINDING
# ============================================================================

def create_env():
    grid = np.zeros((20, 20), dtype=int)

    # Create complex wall patterns to make pathfinding challenging
    # Vertical walls
    grid[3:17, 5] = 1
    grid[3:15, 10] = 1
    grid[5:18, 15] = 1

    # Horizontal walls
    grid[6, 1:10] = 1
    grid[12, 6:16] = 1
    grid[16, 10:19] = 1

    # Additional obstacles
    grid[8:11, 7:9] = 1
    grid[2:5, 12:14] = 1
    grid[14:17, 2:4] = 1

    # Create openings (gaps in walls for paths)
    grid[10, 5] = 0
    grid[8, 10] = 0
    grid[12, 15] = 0
    grid[12, 10] = 0

    # Place Robot starting position
    robot_pos = (1, 1)

    # Place 5 Victims in strategic positions
    victims = {
        1: (2, 18),  # Victim 1 - far right
        2: (9, 3),  # Victim 2 - left side
        3: (18, 8),  # Victim 3 - bottom center
        4: (4, 8),  # Victim 4 - top center
        5: (15, 17)  # Victim 5 - bottom right
    }

    return grid, robot_pos, victims


def bfs(grid, start, target):
    rows, cols = grid.shape

    # Check if start or goal is on a wall
    if grid[start] == 1 or grid[target] == 1:
        return float('inf'), []

    # If start and goal are the same
    if start == target:
        return 0, [start]

    # BFS initialization
    queue = deque([(start, [start])])  # (position, path)
    visited = set([start])

    # Four possible movements: up, down, left, right
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    while queue:
        current_pos, path = queue.popleft()

        # Explore all neighbors
        for dr, dc in directions:
            new_row = current_pos[0] + dr
            new_col = current_pos[1] + dc
            new_pos = (new_row, new_col)

            # Check boundaries
            if 0 <= new_row < rows and 0 <= new_col < cols:
                # Check if not wall and not visited
                if grid[new_pos] == 0 and new_pos not in visited:
                    new_path = path + [new_pos]

                    # Check if we reached the goal
                    if new_pos == target:
                        return len(new_path) - 1, new_path

                    queue.append((new_pos, new_path))
                    visited.add(new_pos)

    # No path found
    return float('inf'), []


def build_distance_matrix(grid, robot_pos, victims):
    # Create a list of all locations: [Robot, V1, V2, V3, V4, V5]
    locations = [robot_pos] + [victims[i] for i in range(1, 6)]
    n = len(locations)

    # Initialize distance matrix
    distance_matrix = np.zeros((n, n), dtype=float)
    path_dict = {}

    print("\n" + "=" * 60)
    print("BUILDING DISTANCE MATRIX USING BFS")
    print("=" * 60)

    # Calculate distances between all pairs
    for i in range(n):
        for j in range(n):
            if i == j:
                distance_matrix[i][j] = 0
                path_dict[(i, j)] = [locations[i]]
            else:
                dist, path = bfs(grid, locations[i], locations[j])
                distance_matrix[i][j] = dist
                path_dict[(i, j)] = path

                # Print distance info
                loc_i = "Robot" if i == 0 else f"V{i}"
                loc_j = "Robot" if j == 0 else f"V{j}"
                print(f"Distance from {loc_i} to {loc_j}: {dist} steps")

    print("=" * 60 + "\n")

    return distance_matrix, path_dict

