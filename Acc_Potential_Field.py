import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle, Circle
import matplotlib.colors as mcolors

# Constants
N = 5  # Number of robots
DIM = 2  # 2D plane (x, y)
SAFE_DISTANCE = 0.5
NUM_FRAMES = 200  # Increased for slower dynamics
DISTANCE_THRESHOLD = 0.2
VELOCITY_THRESHOLD = 0.09
MAX_ACCELERATION = 2.0  # Maximum acceleration magnitude
DT = 0.1  # Time step for integration
DAMPING = 1  # Damping coefficient for velocity

# Weights for different forces
GOAL_WEIGHT = 0.7
COLLISION_WEIGHT = 1.0
OBSTACLE_WEIGHT = 1.5
OBSTACLE_SAFE_DISTANCE = 0.5

# World setup
WORLD_SIZE = 10

# Generate distinct colors for agents
colors = list(mcolors.TABLEAU_COLORS.values())[:N]

# Static obstacles (rectangles defined by [x, y, width, height])
static_obstacles = [
    [3, 2.5, 1, .5],
    [7, 4, 2, 1],
    [3, 7, 1, 2]
]

# Dynamic obstacles (circles defined by [x, y, radius, velocity_x, velocity_y])
dynamic_obstacles = [
    [4, 4, 0.5, 0.03, 0.015],
    [7, 3, 0.4, -0.02, 0.01]
]

def check_static_obstacle_collision(point, margin=0):
    """Check if a point collides with any static obstacle"""
    for obs in static_obstacles:
        x, y = point
        if (x + margin > obs[0] and x - margin < obs[0] + obs[2] and 
            y + margin > obs[1] and y - margin < obs[1] + obs[3]):
            return True
    return False

def check_dynamic_obstacle_collision(point, margin=0):
    """Check if a point collides with any dynamic obstacle"""
    for obs in dynamic_obstacles:
        distance = np.linalg.norm(np.array([point[0] - obs[0], point[1] - obs[1]]))
        if distance < obs[2] + margin:
            return True
    return False

def generate_safe_position(position=None):
    """Generate a random position that doesn't collide with obstacles"""
    margin = max(SAFE_DISTANCE, OBSTACLE_SAFE_DISTANCE)
    while True:
        # position = np.random.rand(2) * (WORLD_SIZE - 4) + 2
        if (not check_static_obstacle_collision(position, margin) and
            not check_dynamic_obstacle_collision(position, margin)):
            return position

# Initialize robot states
robots = np.zeros((N, DIM))  # Positions
velocities = np.zeros((N, DIM))  # Velocities
destinations = np.zeros((N, DIM))  # Goal positions

# Initialize positions and destinations
# for i in range(N):
#     robots[i] = generate_safe_position()
#     destinations[i] = generate_safe_position()

pos_vec = [[2, 2], [4, 2], [6, 2], [2, 4], [2, 6]]
dest_vec = [[8, 8], [1, 5], [5, 7], [8, 1], [6, 3]] 
# robots = np.array(pos_vec)
# destinations = np.array(dest_vec)

for i in range(N):
    robots[i] = generate_safe_position(position=np.array(pos_vec[i]))
    destinations[i] = generate_safe_position(position=np.array(dest_vec[i]))

def update_dynamic_obstacles():
    """Update positions of dynamic obstacles"""
    for obstacle in dynamic_obstacles:
        obstacle[0] += obstacle[3]
        obstacle[1] += obstacle[4]
        
        if obstacle[0] - obstacle[2] < 0 or obstacle[0] + obstacle[2] > WORLD_SIZE:
            obstacle[3] *= -1
        if obstacle[1] - obstacle[2] < 0 or obstacle[1] + obstacle[2] > WORLD_SIZE:
            obstacle[4] *= -1

def get_obstacle_avoidance_force(position):
    """Calculate repulsive force from obstacles"""
    total_force = np.zeros(2)
    
    # Static obstacles
    for obs in static_obstacles:
        closest_x = max(obs[0], min(position[0], obs[0] + obs[2]))
        closest_y = max(obs[1], min(position[1], obs[1] + obs[3]))
        closest_point = np.array([closest_x, closest_y])
        
        vector_to_obstacle = position - closest_point
        distance = np.linalg.norm(vector_to_obstacle)
        
        if distance < OBSTACLE_SAFE_DISTANCE:
            force_magnitude = (OBSTACLE_SAFE_DISTANCE - distance) ** 2
            if distance > 0:
                total_force += (vector_to_obstacle / distance) * force_magnitude
    
    # Dynamic obstacles
    for obs in dynamic_obstacles:
        obstacle_pos = np.array([obs[0], obs[1]])
        vector_to_obstacle = position - obstacle_pos
        distance = np.linalg.norm(vector_to_obstacle)
        
        if distance < OBSTACLE_SAFE_DISTANCE + obs[2]:
            force_magnitude = (OBSTACLE_SAFE_DISTANCE + obs[2] - distance) ** 2
            if distance > 0:
                total_force += (vector_to_obstacle / distance) * force_magnitude
    
    return total_force

def compute_acceleration(robot_positions, velocities, i):
    """Compute acceleration for robot i based on potential field"""
    current_pos = robot_positions[i]
    current_vel = velocities[i]
    
    # Goal attraction with distance-dependent weight
    direction_to_goal = destinations[i] - current_pos
    distance_to_goal = np.linalg.norm(direction_to_goal)
    
    # Adjust goal weight based on distance
    adaptive_goal_weight = GOAL_WEIGHT
    adaptive_damping = DAMPING
    
    if distance_to_goal < SAFE_DISTANCE:
        # Reduce goal attraction and increase damping when close to goal
        adaptive_goal_weight *= (distance_to_goal / SAFE_DISTANCE)
        adaptive_damping *= (1 + (SAFE_DISTANCE - distance_to_goal) / SAFE_DISTANCE)
    
    if distance_to_goal > 0:
        goal_direction = direction_to_goal / distance_to_goal
    else:
        goal_direction = np.zeros(2)
    
    # Robot-robot avoidance
    repulsive_force = np.zeros(2)
    for j in range(N):
        if i != j:
            vector_to_robot = current_pos - robot_positions[j]
            distance = np.linalg.norm(vector_to_robot)
            
            if distance < SAFE_DISTANCE:
                force_magnitude = (SAFE_DISTANCE - distance) ** 2 / SAFE_DISTANCE
                if distance > 0:
                    repulsive_force += (vector_to_robot / distance) * force_magnitude
    
    # Obstacle avoidance
    obstacle_force = get_obstacle_avoidance_force(current_pos)
    
    # Normalize forces
    if np.linalg.norm(repulsive_force) > 0:
        repulsive_force = repulsive_force / np.linalg.norm(repulsive_force)
    if np.linalg.norm(obstacle_force) > 0:
        obstacle_force = obstacle_force / np.linalg.norm(obstacle_force)
    
    # Combine forces with adaptive weights
    acceleration = (adaptive_goal_weight * goal_direction +
                   COLLISION_WEIGHT * repulsive_force +
                   OBSTACLE_WEIGHT * obstacle_force -
                   adaptive_damping * current_vel)
    
    # Additional velocity damping when very close to goal
    if distance_to_goal < DISTANCE_THRESHOLD:
        acceleration -= current_vel * 2  # Extra damping term
    
    # Limit acceleration magnitude
    acc_magnitude = np.linalg.norm(acceleration)
    if acc_magnitude > MAX_ACCELERATION:
        acceleration = acceleration * MAX_ACCELERATION / acc_magnitude
    
    return acceleration

# Set up the plot
fig, ax = plt.subplots(figsize=(16, 10))  # Increased figure width to accommodate legend

# Create dummy elements for legend
legend_elements = [
    # Add individual robot entries
    *[plt.Line2D([0], [0], marker='o', color=colors[i], 
                 label=f'Robot {i+1}', markersize=10, linestyle='none') 
      for i in range(N)],
    # Add individual destination entries
    *[plt.Line2D([0], [0], marker='x', color=colors[i], 
                 label=f'Destination {i+1}', markersize=10, linestyle='none') 
      for i in range(N)],
    # Add other elements
    Rectangle((0, 0), 1, 1, facecolor='gray', alpha=0.5, label='Static Obstacles'),
    plt.scatter([], [], c='red', alpha=0.5, s=200, label='Dynamic Obstacles')  # Changed to scatter point
]

# Add legend with increased font size
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
         fontsize=10, ncol=1)

# Plot destinations
for i in range(N):
    ax.scatter(destinations[i, 0], destinations[i, 1], c=colors[i], marker='x', s=100)
    ax.text(destinations[i, 0] + 0.2, destinations[i, 1] + 0.2, f'D{i+1}', color=colors[i])

# Initialize visualization elements
robot_dots = [ax.plot([], [], 'o', color=colors[i], markersize=10)[0] for i in range(N)]
robot_numbers = [ax.text([], [], f'R{i+1}', color=colors[i]) for i in range(N)]
velocity_arrows = [ax.quiver([], [], [], [], color=colors[i], scale=20) for i in range(N)]
lines = [ax.plot([], [], '--', color=colors[i], alpha=0.3)[0] for i in range(N)]
safety_circles = [plt.Circle((0, 0), SAFE_DISTANCE/2, fill=False, linestyle='--', 
                           color=colors[i], alpha=0.3) for i in range(N)]

# Add obstacles to plot
static_patches = [ax.add_patch(Rectangle((obs[0], obs[1]), obs[2], obs[3], 
                                       facecolor='gray', alpha=0.5)) 
                 for obs in static_obstacles]
dynamic_patches = [ax.add_patch(Circle((obs[0], obs[1]), obs[2], 
                                     facecolor='red', alpha=0.5)) 
                  for obs in dynamic_obstacles]

for circle in safety_circles:
    ax.add_artist(circle)

ax.set_xlim(0, WORLD_SIZE)
ax.set_ylim(0, WORLD_SIZE)
ax.set_xlabel('X Position', fontsize=24)
ax.set_ylabel('Y Position', fontsize=24)
ax.set_title('Multi-Robot Navigation with Dynamic Control', fontsize=28)
ax.tick_params(axis='both', which='major', labelsize=18)  # Increase tick label size
ax.grid(True)
ax.set_aspect('equal')

# Save initial layout
for i in range(N):
    robot_dots[i].set_data([robots[i, 0]], [robots[i, 1]])
    robot_numbers[i].set_position((robots[i, 0] - 0.2, robots[i, 1] - 0.2))
    safety_circles[i].center = robots[i]

# Adjust figure size for better PDF output
fig.set_size_inches(12, 10)
fig.savefig('pot_initial_layout.pdf', format='pdf', bbox_inches='tight', dpi=300)

position_history = [[] for _ in range(N)]
velocity_history = [[] for _ in range(N)]
accelaration_history = [[] for _ in range(N)]

def init():
    for dot, text, arrow in zip(robot_dots, robot_numbers, velocity_arrows):
        dot.set_data([], [])
        text.set_position((0, 0))
        arrow.set_offsets(np.array([0, 0]))
        arrow.set_UVC(0, 0)
    for line in lines:
        line.set_data([], [])
    for circle in safety_circles:
        circle.center = (0, 0)
    return robot_dots + robot_numbers + velocity_arrows + lines + safety_circles + dynamic_patches

def update(frame):
    global robots, velocities
    
    # Update dynamic obstacles
    update_dynamic_obstacles()
    
    # Update robot states
    for i in range(N):
        # Compute acceleration
        acceleration = compute_acceleration(robots, velocities, i)
        
        # Update velocity (semi-implicit Euler integration)
        new_velocity = velocities[i] + acceleration * DT
        
        # Update position
        new_position = robots[i] + new_velocity * DT
        
        # Check if new position is valid
        if not check_static_obstacle_collision(new_position, margin=0.2):
            robots[i] = new_position
            velocities[i] = new_velocity
            
        position_history[i].append(robots[i].copy())
        velocity_history[i].append(velocities[i].copy())
        accelaration_history[i].append(acceleration.copy())

    np.save('acc_robot_positions.npy', position_history)
    np.save('acc_robot_velocities.npy', velocity_history)
    np.save('acc_robot_acceleration.npy', accelaration_history)
    
    # Save data at the end of simulation
    if frame == NUM_FRAMES - 1:
        np.save('acc_robot_positions.npy', np.array(position_history))
        np.save('acc_robot_velocities.npy', np.array(velocity_history))
        plot_robot_data()
    
    # Update visualization
    for i, (dot, number, arrow) in enumerate(zip(robot_dots, robot_numbers, velocity_arrows)):
        dot.set_data([robots[i, 0]], [robots[i, 1]])
        number.set_position((robots[i, 0] - 0.2, robots[i, 1] - 0.2))
        arrow.set_offsets(robots[i])
        arrow.set_UVC(velocities[i, 0], velocities[i, 1])
    
    # Update safety circles and trajectories
    for i, (circle, line) in enumerate(zip(safety_circles, lines)):
        circle.center = robots[i]
        if position_history[i]:
            positions = np.array(position_history[i])
            line.set_data(positions[:, 0], positions[:, 1])
    
    # Update dynamic obstacle visualizations
    for obstacle, patch in zip(dynamic_obstacles, dynamic_patches):
        patch.center = (obstacle[0], obstacle[1])
    
    # Check if all robots reached their goals
    distances_to_goals = np.linalg.norm(robots - destinations, axis=1)
    velocities_magnitude = np.linalg.norm(velocities, axis=1)
    if (np.all(distances_to_goals <= DISTANCE_THRESHOLD) and 
        np.all(velocities_magnitude <= VELOCITY_THRESHOLD)) or frame >= NUM_FRAMES - 1:
        anim.event_source.stop()
    
    return robot_dots + robot_numbers + velocity_arrows + lines + safety_circles + dynamic_patches

def plot_robot_data():
    """Plot robot positions and velocities over time"""
    positions = np.array(position_history)  # Shape: (N, timesteps, 2)
    velocities = np.array(velocity_history)
    time = np.arange(len(position_history[0])) * DT

    # Create position plots
    fig_pos, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    fig_pos.suptitle('Robot Positions Over Time')
    
    for i in range(N):
        ax1.plot(time, positions[i, :, 0], label=f'Robot {i}', color=colors[i])
        ax2.plot(time, positions[i, :, 1], label=f'Robot {i}', color=colors[i])
    
    ax1.set_ylabel('X Position')
    ax2.set_ylabel('Y Position')
    ax2.set_xlabel('Time (s)')
    ax1.grid(True)
    ax2.grid(True)
    ax1.legend()
    
    # Create velocity plots
    fig_vel, (ax3, ax4) = plt.subplots(2, 1, figsize=(14, 8))
    fig_vel.suptitle('Robot Velocities Over Time')
    
    for i in range(N):
        ax3.plot(time, velocities[i, :, 0], label=f'Robot {i}', color=colors[i])
        ax4.plot(time, velocities[i, :, 1], label=f'Robot {i}', color=colors[i])
    
    ax3.set_ylabel('X Velocity')
    ax4.set_ylabel('Y Velocity')
    ax4.set_xlabel('Time (s)')
    ax3.grid(True)
    ax4.grid(True)
    ax3.legend()
    
    # plt.show()

anim = FuncAnimation(fig, update, frames=NUM_FRAMES, init_func=init, 
                    blit=True, interval=20)

writervideo = animation.FFMpegWriter(fps=10) 
anim.save('pot_anim.mp4', writer=writervideo) 

plt.show()
plot_robot_data()