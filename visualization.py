import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def visualize_solution(grid, robot_pos, victims, best_order, path_dict,
                       distance_matrix, ga, save_path='rescue_mission_solution.png'):
    """
    Visualizes the grid environment and the optimal rescue path.
    """
    fig, ax1 = plt.subplots(figsize=(9, 9))

    # ========== LEFT PLOT: GRID WITH OPTIMAL PATH ==========
    ax1.set_xlim(-0.5, grid.shape[1] - 0.5)
    ax1.set_ylim(-0.5, grid.shape[0] - 0.5)
    ax1.set_aspect('equal')
    ax1.invert_yaxis()
    ax1.set_title('Rescue Mission: Optimal Path (GA Solution)', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Column')
    ax1.set_ylabel('Row')
    ax1.grid(True, alpha=0.3)

    # Draw walls
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if grid[i, j] == 1:
                rect = Rectangle((j - 0.5, i - 0.5), 1, 1,
                                 linewidth=0, facecolor='black', alpha=0.8)
                ax1.add_patch(rect)

    # Draw Robot
    ax1.plot(robot_pos[1], robot_pos[0], 'gs', markersize=20,
             label='Robot Start', markeredgecolor='black', markeredgewidth=2)
    ax1.text(robot_pos[1], robot_pos[0], 'R', ha='center', va='center',
             fontsize=12, fontweight='bold', color='white')

    # Draw Victims
    colors = ['red', 'blue', 'purple', 'orange', 'brown']
    for victim_id in range(1, 6):
        pos = victims[victim_id]
        ax1.plot(pos[1], pos[0], 'o', color=colors[victim_id - 1],
                 markersize=18, label=f'Victim {victim_id}',
                 markeredgecolor='black', markeredgewidth=2)
        ax1.text(pos[1], pos[0], f'V{victim_id}', ha='center', va='center',
                 fontsize=10, fontweight='bold', color='white')

    # Draw optimal path
    current_idx = 0  # Start from robot
    total_distance = 0

    for step_num, victim_id in enumerate(best_order, 1):
        next_idx = victim_id
        path = path_dict[(current_idx, next_idx)]

        if len(path) > 1:
            path_array = np.array(path)
            ax1.plot(path_array[:, 1], path_array[:, 0],
                     'y-', linewidth=3, alpha=0.7, zorder=1)

            # Add arrow to show direction
            mid_point = len(path) // 2
            if mid_point + 1 < len(path):
                ax1.annotate('', xy=(path[mid_point + 1][1], path[mid_point + 1][0]),
                             xytext=(path[mid_point][1], path[mid_point][0]),
                             arrowprops=dict(arrowstyle='->', color='yellow',
                                             lw=2, alpha=0.8))

        segment_distance = distance_matrix[current_idx][next_idx]
        total_distance += segment_distance
        current_idx = next_idx

    ax1.legend(loc='upper left', fontsize=9)
    ax1.text(0.02, 0.02, f'Total Distance: {total_distance:.0f} steps\nVisit Order: {best_order}',
             transform=ax1.transAxes, fontsize=11, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved as '{save_path}'")
    plt.show()
