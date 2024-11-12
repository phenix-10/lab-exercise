import numpy as np
import random
import matplotlib.pyplot as plt
from queue import Queue

# Environment Setup
GRID_SIZE = 10  # 10x10 grid
NUM_OBSTACLES = 15

# Initialize the grid
grid = np.zeros((GRID_SIZE, GRID_SIZE))

# Place obstacles randomly
for _ in range(NUM_OBSTACLES):
    x, y = random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1)
    grid[x][y] = 1  # 1 indicates an obstacle

# Place the target randomly
target_position = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
grid[target_position] = 2  # 2 indicates the target

# Visualize initial grid setup
plt.imshow(grid, cmap="coolwarm", interpolation="nearest")
plt.title("2D Grid Environment")
plt.show()


# Agent Definition
class Agent:
    def __init__(self, grid, start_position=(0, 0)):
        self.grid = grid
        self.position = start_position  # Starting position
        self.grid_size = len(grid)

    # Perceive the immediate surroundings
    def perceive(self):
        x, y = self.position
        perception = {
            "up": self.grid[x - 1][y] if x > 0 else None,
            "down": self.grid[x + 1][y] if x < self.grid_size - 1 else None,
            "left": self.grid[x][y - 1] if y > 0 else None,
            "right": self.grid[x][y + 1] if y < self.grid_size - 1 else None
        }
        return perception

    # Action: Move the agent based on the direction chosen
    def move(self, direction):
        x, y = self.position
        if direction == "up" and x > 0:
            self.position = (x - 1, y)
        elif direction == "down" and x < self.grid_size - 1:
            self.position = (x + 1, y)
        elif direction == "left" and y > 0:
            self.position = (x, y - 1)
        elif direction == "right" and y < self.grid_size - 1:
            self.position = (x, y + 1)


# BFS Pathfinding Algorithm
def bfs(grid, start, target):
    queue = Queue()
    queue.put((start, [start]))  # Queue holds tuples of (current_position, path)
    visited = set()
    visited.add(start)

    while not queue.empty():
        position, path = queue.get()
        x, y = position

        if position == target:
            return path  # Path to target found

        # Possible moves
        moves = {
            "up": (x - 1, y),
            "down": (x + 1, y),
            "left": (x, y - 1),
            "right": (x, y + 1)
        }

        for move, (new_x, new_y) in moves.items():
            if 0 <= new_x < len(grid) and 0 <= new_y < len(grid[0]) and \
                    grid[new_x][new_y] != 1 and (new_x, new_y) not in visited:  # Check boundary, obstacles, and visited
                queue.put(((new_x, new_y), path + [(new_x, new_y)]))
                visited.add((new_x, new_y))

    return None  # If target is not reachable


# Initialize Agent and Pathfinding
agent = Agent(grid)
start_position = agent.position

# Run BFS to get the path to the target
path = bfs(grid, start_position, target_position)

# Simulation of the Agent Moving along the Path
if path:
    print("Path to target found:", path)
    for step in path:
        agent.position = step
        print(f"Agent moves to: {agent.position}")

        # Visualization of agent's movement
        plt.imshow(grid, cmap="coolwarm", interpolation="nearest")
        plt.plot(agent.position[1], agent.position[0], 'bo')  # 'bo' marks the agent
        plt.pause(0.5)  # Pause to visualize each step
    print("Target reached!")
else:
    print("No path to target found.")
