import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import matplotlib.animation as animation

class Robot:
    def __init__(self, pos, goal):
        self.pos = np.array(pos)
        self.goal = np.array(goal)
        self.trajectory = [self.pos.copy()]
        self.velocities = []  # Store velocity history
        self.times = []      # Store time history

class DynamicObstacle:
    def __init__(self, pos, velocity):
        self.pos = np.array(pos)
        self.velocity = np.array(velocity)
        self.trajectory = [self.pos.copy()]
    
    def update(self, dt):
        self.pos += self.velocity * dt
        # Bounce off boundaries
        for i in range(2):
            if self.pos[i] <= 0 or self.pos[i] >= 10:
                self.velocity[i] *= -1
                self.pos[i] = np.clip(self.pos[i], 0, 10)
        self.trajectory.append(self.pos.copy())

def optimize_velocity(robot, other_robots, static_obstacles, dynamic_obstacles, v_max, alpha, dt):
    n = len(robot.pos)  # dimension (2D = 2)
    
    # Objective: minimize 1/2 ||p_i - g_i||^2
    P = matrix(np.eye(n) * 2.0)
    q = matrix(-2.0 * (robot.goal - robot.pos))
    
    # Basic velocity constraints
    G_vel = np.vstack([np.eye(n), -np.eye(n)])
    h_vel = np.ones(2*n) * v_max
    
    G_list = [G_vel]
    h_list = [h_vel]
    
    # Collision avoidance constraints
    for other in other_robots:
        diff = robot.pos - other.pos
        dist = np.linalg.norm(diff)
        if dist < 2*alpha and dist > 1e-6:  # Add small threshold to avoid division by zero
            # Normalize direction vector
            diff_normalized = diff / dist
            G_list.append(-diff_normalized.reshape(1,-1))
            h_list.append(np.array([-0.1]))  # Small positive velocity in avoidance direction
    
    for obs in static_obstacles:
        diff = robot.pos - obs
        dist = np.linalg.norm(diff)
        if dist < 2*alpha and dist > 1e-6:
            diff_normalized = diff / dist
            G_list.append(-diff_normalized.reshape(1,-1))
            h_list.append(np.array([-0.1]))
    
    # Dynamic obstacles (with velocity prediction)
    for obs in dynamic_obstacles:
        predicted_pos = obs.pos + obs.velocity * dt
        diff = robot.pos - predicted_pos
        dist = np.linalg.norm(diff)
        if dist < 2*alpha and dist > 1e-6:
            diff_normalized = diff / dist
            G_list.append(-diff_normalized.reshape(1,-1))
            h_list.append(np.array([-0.2]))  # Larger margin for moving obstacles
    
    # Combine all constraints
    G = matrix(np.vstack(G_list))
    h = matrix(np.hstack(h_list))
    
    try:
        # Solve optimization problem
        solvers.options['show_progress'] = False
        solvers.options['maxiters'] = 100
        solvers.options['abstol'] = 1e-7
        solvers.options['reltol'] = 1e-6
        solvers.options['feastol'] = 1e-7
        
        solution = solvers.qp(P, q, G, h)
        
        if solution['status'] == 'optimal':
            return np.array(solution['x']).flatten()
        else:
            # If optimization fails, move away from closest robot/obstacle
            closest_dist = float('inf')
            move_direction = np.zeros(2)
            
            for other in other_robots:
                diff = robot.pos - other.pos
                dist = np.linalg.norm(diff)
                if dist < closest_dist:
                    closest_dist = dist
                    move_direction = diff / (dist + 1e-6)
            
            for obs in static_obstacles:
                diff = robot.pos - obs
                dist = np.linalg.norm(diff)
                if dist < closest_dist:
                    closest_dist = dist
                    move_direction = diff / (dist + 1e-6)
            
            return move_direction * v_max
            
    except Exception as e:
        # Fallback behavior: move towards goal if far from obstacles
        direction = robot.goal - robot.pos
        dist = np.linalg.norm(direction)
        if dist > 1e-6:
            return (direction / dist) * v_max * 0.5
        return np.zeros(n)

def simulate_robots(n_robots, static_obstacles, dynamic_obstacles, v_max, alpha, dt, simulation_time):
    # Initialize robots with better spacing
    robots = []
    # np.random.seed(0)
    
    # Create a grid of initial positions
    grid_size = int(np.ceil(np.sqrt(n_robots)))
    spacing = 10.0 / (grid_size + 1)
    
    pos = [[1, 1], [1, 9], [9, 1], [9, 9], [5, 5]]
    goal = [[5.5, 6.8], [7, 3], [1, 4], [1, 2], [8, 3]]

    pos = np.array(pos, dtype=np.float64)
    goal = np.array(goal, dtype=np.float64)

    for i in range(n_robots):
        robots.append(Robot(pos[i], goal[i]))

    # for i in range(n_robots):
    #     row = i // grid_size
    #     col = i % grid_size
    #     pos = np.array([spacing * (col + 1), spacing * (row + 1)])
    #     # Generate goals away from obstacles
    #     while True:
    #         goal = np.random.rand(2) * 10
    #         valid = True
    #         for obs in static_obstacles:
    #             if np.linalg.norm(goal - obs) < 2*alpha:
    #                 valid = False
    #                 break
    #         if valid:
    #             break
    #     robots.append(Robot(pos, goal))
    
    times = np.arange(0, simulation_time, dt)
    n_steps = len(times)
    
    # Initialize arrays with correct shape (n_robots, timesteps, 2)
    all_positions = np.zeros((n_robots, n_steps, 2))
    all_velocities = np.zeros((n_robots, n_steps, 2))
    all_obstacle_positions = np.zeros((len(dynamic_obstacles), n_steps, 2))
    
    for t_idx, t in enumerate(times):
        # Update dynamic obstacles
        for obs_idx, obs in enumerate(dynamic_obstacles):
            obs.update(dt)
            all_obstacle_positions[obs_idx, t_idx] = obs.pos
        
        # Update robots
        for i, robot in enumerate(robots):
            other_robots = robots[:i] + robots[i+1:]
            velocity = optimize_velocity(robot, other_robots, static_obstacles, dynamic_obstacles, v_max, alpha, dt)
            robot.pos += velocity * dt
            robot.trajectory.append(robot.pos.copy())
            robot.velocities.append(velocity)
            robot.times.append(t)
            
            # Store data in arrays with correct indexing
            all_positions[i, t_idx] = robot.pos
            all_velocities[i, t_idx] = velocity
    
    # Save data with new shape
    timestamp = np.datetime64('now').astype(str).replace(':', '-')
    np.save(f'robot_positions_{timestamp}.npy', all_positions)
    np.save(f'robot_velocities_{timestamp}.npy', all_velocities)
    np.save(f'obstacle_positions_{timestamp}.npy', all_obstacle_positions)
    np.save(f'simulation_times_{timestamp}.npy', times)
    
    return robots, dynamic_obstacles, all_positions, all_velocities, times

def animate(frame, robots, static_obstacles, dynamic_obstacles, lines, points, trails, targets, obstacle_points, velocity_arrows, safety_circles):
    for line, point, trail, target, robot, arrow, circle, color in zip(lines, points, trails, targets, robots, velocity_arrows, safety_circles, get_custom_colors(len(robots))):
        traj = np.array(robot.trajectory[:frame+1])
        current_pos = traj[-1]
        if frame < len(robot.velocities):
            current_vel = robot.velocities[frame]
            # Update velocity arrow with smaller size
            arrow.set_offsets(current_pos)
            scaled_vel = current_vel * 2.0  # Reduced scaling factor
            arrow.set_UVC(scaled_vel[0], scaled_vel[1])
        
        # Update safety radius circle position
        circle.center = current_pos
        
        # Update other elements
        line.set_data([robot.pos[0], robot.goal[0]], [robot.pos[1], robot.goal[1]])
        point.set_data([current_pos[0]], [current_pos[1]])
        trail.set_data(traj[:,0], traj[:,1])
        target.set_data([robot.goal[0]], [robot.goal[1]])
    
    # Update dynamic obstacles
    for obs_point, obs in zip(obstacle_points, dynamic_obstacles):
        traj = np.array(obs.trajectory[:frame+1])
        obs_point.set_data([traj[-1,0]], [traj[-1,1]])
    
    return lines + points + trails + targets + obstacle_points + velocity_arrows + safety_circles

def get_custom_colors(n_robots):
    """Create custom colors for robots, ensuring the last one isn't red"""
    colors = []
    base_colors = ['blue', 'green', 'purple', 'orange', 'cyan']  # Changed last color from red to cyan
    for i in range(n_robots):
        colors.append(base_colors[i % len(base_colors)])
    return colors

def plot_robot_data(robots, times):
    n_robots = len(robots)
    colors = get_custom_colors(n_robots)  # Replace rainbow colors with custom colors
    
    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    for i, (robot, color) in enumerate(zip(robots, colors)):
        velocities = np.array(robot.velocities)
        trajectory = np.array(robot.trajectory)
        
        # Plot velocity magnitudes
        speed = np.linalg.norm(velocities, axis=1)
        ax1.plot(robot.times, speed, color=color, label=f'Robot {i+1}')
        
        # Plot x position over time
        ax2.plot(robot.times, trajectory[:-1, 0], color=color, label=f'Robot {i+1}')
        
        # Plot y position over time
        ax3.plot(robot.times, trajectory[:-1, 1], color=color, label=f'Robot {i+1}')
        
        # Plot x-y trajectories
        ax4.plot(trajectory[:, 0], trajectory[:, 1], color=color, label=f'Robot {i+1}')
        ax4.plot(robot.goal[0], robot.goal[1], '*', color=color, markersize=10)
    
    # Configure subplots
    ax1.set_xlabel('Time [s]')
    ax1.set_ylabel('Speed [m/s]')
    ax1.set_title('Robot Speeds over Time')
    ax1.grid(True)
    ax1.legend()
    
    ax2.set_xlabel('Time [s]')
    ax2.set_ylabel('X Position [m]')
    ax2.set_title('X Positions over Time')
    ax2.grid(True)
    ax2.legend()
    
    ax3.set_xlabel('Time [s]')
    ax3.set_ylabel('Y Position [m]')
    ax3.set_title('Y Positions over Time')
    ax3.grid(True)
    ax3.legend()
    
    ax4.set_xlabel('X Position [m]')
    ax4.set_ylabel('Y Position [m]')
    ax4.set_title('Robot Trajectories')
    ax4.grid(True)
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig('robot_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# def plot_initial_layout(robots, static_obstacles, dynamic_obstacles):
#     fig = plt.figure(figsize=(10, 8))  # Changed figure size for better aspect ratio
#     ax = fig.add_subplot(111)
#     ax.set_xlim(0, 10)
#     ax.set_ylim(0, 10)
#     ax.set_aspect('equal')  # Force equal aspect ratio
#     ax.grid(True, linestyle='--', alpha=0.7)
    
#     # Plot static obstacles with different color
#     for obs in static_obstacles:
#         circle = plt.Circle(obs, 0.2, color='darkgray', fill=True, alpha=0.7)
#         ax.add_patch(circle)
    
#     # Plot robots and their goals
#     colors = get_custom_colors(len(robots))
#     for i, (robot, color) in enumerate(zip(robots, colors)):
#         # Plot robot position
#         ax.plot(robot.pos[0], robot.pos[1], 'o', color=color, markersize=10, label=f'Robot {i+1}')
#         # Plot goal position
#         ax.plot(robot.goal[0], robot.goal[1], '*', color=color, markersize=15, label=f'Goal {i+1}')
    
#     # Plot dynamic obstacles with velocity vectors
#     for obs in dynamic_obstacles:
#         ax.plot(obs.pos[0], obs.pos[1], 'ro', markersize=15, alpha=0.7)
#         ax.arrow(obs.pos[0], obs.pos[1], 
#                 obs.velocity[0], obs.velocity[1],
#                 color='red', width=0.05, alpha=0.5)
    
#     ax.set_title('Initial Environment Layout')
    
#     # Create legend with all elements and place it outside
#     handles, labels = ax.get_legend_handles_labels()
#     ax.plot([], [], 'ro', markersize=15, alpha=0.7, label='Dynamic Obstacle')
#     ax.plot([], [], 'o', color='darkgray', markersize=15, alpha=0.7, label='Static Obstacle')
#     ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)  # Place legend outside
    
#     plt.tight_layout()  # Adjust layout to make room for legend
#     plt.subplots_adjust(right=0.85)
#     plt.savefig('opt_initial_layout.pdf', format='pdf', dpi=300, bbox_inches='tight')
#     # plt.close()

def plot_initial_layout(robots, static_obstacles, dynamic_obstacles):
    plt.rcParams.update({'font.size': 20})  # Increase base font size
    
    fig = plt.figure(figsize=(10, 8))  # Changed figure size for better aspect ratio
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')  # Force equal aspect ratio
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Plot static obstacles with different color
    for obs in static_obstacles:
        circle = plt.Circle(obs, 0.2, color='darkgray', fill=True, alpha=0.7)
        ax.add_patch(circle)
    
    # Plot robots and their goals
    colors = get_custom_colors(len(robots))
    for i, (robot, color) in enumerate(zip(robots, colors)):
        # Plot robot position
        ax.plot(robot.pos[0], robot.pos[1], 'o', color=color, markersize=10, label=f'Robot {i+1}')
        # Plot goal position
        ax.plot(robot.goal[0], robot.goal[1], '*', color=color, markersize=15, label=f'Goal {i+1}')
    
    # Plot dynamic obstacles with velocity vectors
    for obs in dynamic_obstacles:
        ax.plot(obs.pos[0], obs.pos[1], 'ro', markersize=15, alpha=0.7)
        ax.arrow(obs.pos[0], obs.pos[1], 
                obs.velocity[0], obs.velocity[1],
                color='red', width=0.05, alpha=0.5)
    
    ax.set_title('Initial Environment Layout', fontsize=28)
    ax.set_xlabel('X Position [m]', fontsize=24)
    ax.set_ylabel('Y Position [m]', fontsize=24)
    ax.tick_params(axis='both', which='major', labelsize=18)
    
    # Create legend with all elements and place it outside
    handles, labels = ax.get_legend_handles_labels()
    ax.plot([], [], 'ro', markersize=15, alpha=0.7, label='Dynamic Obstacle')
    ax.plot([], [], 'o', color='darkgray', markersize=15, alpha=0.7, label='Static Obstacle')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1, fontsize=20)  # Place legend outside
    
    plt.tight_layout()  # Adjust layout to make room for legend
    plt.subplots_adjust(right=0.85)
    plt.savefig('opt_initial_layout.pdf', format='pdf', dpi=300, bbox_inches='tight')
    # plt.close()

def main():
    # Simulation parameters
    n_robots = 5
    static_obstacles = [np.array([3., 4.]),
                 np.array([1., 3.])]  # Single obstacle at center
    dynamic_obstacles = [
        DynamicObstacle([1., 3.5], [0.3, 0.2]),
        DynamicObstacle([8., 7.], [-0.2, -0.1])
    ]
    v_max = 0.5
    alpha = 0.2
    dt = 0.1
    simulation_time = 25.0
    
    # Create robots and plot initial layout before simulation
    robots = []
    pos = [[1, 1], [1, 9], [9, 1], [9, 9], [5, 5]]
    goal = [[5.5, 6.8], [7, 3], [1, 4], [1, 2], [8, 3]]
    pos = np.array(pos, dtype=np.float64)
    goal = np.array(goal, dtype=np.float64)
    for i in range(n_robots):
        robots.append(Robot(pos[i], goal[i]))
    
    plot_initial_layout(robots, static_obstacles, dynamic_obstacles)
    
    # Run simulation with modified return values
    robots, dynamic_obstacles, trajectories, velocities, times = simulate_robots(
        n_robots, static_obstacles, dynamic_obstacles, v_max, alpha, dt, simulation_time
    )
    
    # Print data shapes for verification
    print(f"Trajectories shape: {trajectories.shape}")
    print(f"Velocities shape: {velocities.shape}")
    print(f"Times shape: {times.shape}")
    
    # Plot robot data
    plot_robot_data(robots, np.arange(0, simulation_time, dt))
    
    # Create animation with adjusted figure size
    fig = plt.figure(figsize=(14, 8))  # Changed figure size
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_aspect('equal')  # Force equal aspect ratio
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)  # Place grid behind other elements
    
    # Plot obstacles with updated color
    for obs in static_obstacles:
        circle = plt.Circle(obs, alpha, color='darkgray', fill=True, alpha=0.5)
        ax.add_patch(circle)
    
    # Generate colors for each robot
    colors = get_custom_colors(n_robots)  # Replace rainbow colors with custom colors
    
    # Create visualization elements with different colors for each robot
    lines = []
    points = []
    trails = []
    targets = []
    velocity_arrows = []
    safety_circles = []
    
    for color in colors:
        lines.append(ax.plot([], [], '--', color=color, alpha=0.5)[0])  # Lines to goals
        points.append(ax.plot([], [], 'o', color=color, markersize=10)[0])  # Robot positions
        trails.append(ax.plot([], [], '-', color=color, alpha=0.3)[0])  # Trajectories
        targets.append(ax.plot([], [], '*', color=color, markersize=15)[0])  # Goal markers
        
        # Add velocity arrows with smaller size
        velocity_arrows.append(ax.quiver([], [], [], [], 
                                       color=color,
                                       scale=5,  # Larger scale value = shorter arrows
                                       scale_units='inches',
                                       width=0.01,
                                       headwidth=0.51,
                                       headlength=2,
                                       headaxislength=4))
        
        # Add safety radius visualization
        safety_circle = plt.Circle((0, 0), alpha, color=color, fill=True, alpha=0.2)
        ax.add_patch(safety_circle)
        safety_circles.append(safety_circle)
    
    # Add dynamic obstacles to visualization
    obstacle_points = [ax.plot([], [], 'ro', markersize=15, alpha=0.7)[0] 
                      for _ in dynamic_obstacles]
    
    # Add legend with numbered robots, their destinations, and obstacles
    ax.plot([], [], 'ro', markersize=15, alpha=0.7, label='Dynamic Obstacle')
    ax.plot([], [], 'o', color='darkgray', markersize=15, alpha=0.7, label='Static Obstacle')
    for i, color in enumerate(colors):
        ax.plot([], [], 'o', color=color, markersize=10, label=f'Robot {i+1}')
        ax.plot([], [], '*', color=color, markersize=15, label=f'Goal {i+1}')
    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', ncol=1)  # Place legend outside
    
    plt.tight_layout()  # Adjust layout to make room for legend
    plt.subplots_adjust(right=0.85)
    
    frames = len(robots[0].trajectory)
    anim = animation.FuncAnimation(fig, animate, frames=frames,
                                 fargs=(robots, static_obstacles, dynamic_obstacles, 
                                       lines, points, trails, targets, 
                                       obstacle_points, velocity_arrows, safety_circles),
                                 interval=50, blit=True)
    writervideo = animation.FFMpegWriter(fps=10) 
    anim.save('opt_anim.mp4', writer=writervideo) 
    
    plt.title('Multi-Robot Navigation')
    plt.show()

if __name__ == "__main__":
    np.random.seed(1)
    main()
