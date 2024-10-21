import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from queue import Queue, PriorityQueue

def create_complex_maze(dim, loop_probability=0.3):
    # Create a grid filled with walls
    maze = np.ones((dim*2+1, dim*2+1))

    # Define the starting point
    x, y = (0, 0)
    maze[2*x+1, 2*y+1] = 0

    # Initialize the stack with the starting point
    stack = [(x, y)]
    visited = set([(x, y)])

    while len(stack) > 0:
        x, y = stack[-1]

        # Define possible directions
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 0 <= nx < dim and 0 <= ny < dim and (nx, ny) not in visited:
                visited.add((nx, ny))
                maze[2*nx+1, 2*ny+1] = 0  # Make the cell a path
                maze[2*x+1+dx, 2*y+1+dy] = 0  # Knock down wall between current and next
                stack.append((nx, ny))
                break
        else:
            stack.pop()

    # Create an entrance and an exit
    maze[1, 0] = 0
    maze[-2, -1] = 0

    # Randomly remove additional walls to create loopbacks, adding more pathways
    for _ in range(int(loop_probability * dim**2)):
        x = random.randint(1, dim*2-2)
        y = random.randint(1, dim*2-2)
        if maze[x, y] == 1:  # If it's a wall, remove it
            if random.choice([(0, 1), (1, 0), (0, -1), (-1, 0)]):
                maze[x, y] = 0

    return maze





# Heuristic Function for A*
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# BFS Algorithm
def bfs(maze):
    start = (1, 0)
    goal = (maze.shape[0]-2, maze.shape[1]-1)
    queue = Queue()
    queue.put([start])
    visited = set([start])

    while not queue.empty():
        path = queue.get()
        x, y = path[-1]

        if (x, y) == goal:
            return path

        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = x + dx, y + dy
            if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0 and (nx, ny) not in visited:
                queue.put(path + [(nx, ny)])
                visited.add((nx, ny))
    return None

# DFS Algorithm
def dfs(maze):
    start = (1, 0)
    goal = (maze.shape[0]-2, maze.shape[1]-1)
    stack = [(start, [start])]
    visited = set()

    while stack:
        (x, y), path = stack.pop()

        if (x, y) == goal:
            return path

        if (x, y) not in visited:
            visited.add((x, y))

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
                    stack.append(((nx, ny), path + [(nx, ny)]))
    return None

# A* Algorithm
def a_star(maze):
    start = (1, 0)
    goal = (maze.shape[0]-2, maze.shape[1]-1)
    pq = PriorityQueue()
    pq.put((0, [start]))
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    visited = set()

    while not pq.empty():
        _, path = pq.get()
        x, y = path[-1]

        if (x, y) == goal:
            return path

        if (x, y) not in visited:
            visited.add((x, y))

            for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < maze.shape[0] and 0 <= ny < maze.shape[1] and maze[nx, ny] == 0:
                    tentative_g_score = g_score[(x, y)] + 1
                    if (nx, ny) not in g_score or tentative_g_score < g_score[(nx, ny)]:
                        g_score[(nx, ny)] = tentative_g_score
                        f_score[(nx, ny)] = tentative_g_score + heuristic((nx, ny), goal)
                        pq.put((f_score[(nx, ny)], path + [(nx, ny)]))
    return None

# Function to draw and animate the maze
def animate_path(maze, path, ax, color='red'):
    ax.imshow(maze, cmap=plt.cm.binary, interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])

    line, = ax.plot([], [], color=color, linewidth=2)
    
    def init():
        line.set_data([], [])
        return line,

    # Update is called for each step in the path
    def update(frame):
        line.set_data(*zip(*[(p[1], p[0]) for p in path[:frame+1]]))  # update the data
        return line,

    ani = animation.FuncAnimation(ax.figure, update, frames=range(len(path)), init_func=init, blit=True, repeat=False, interval=20)
    return ani

# Main execution
if __name__ == "__main__":
    dim = int(input("Enter the dimension of the maze: "))
    maze = create_complex_maze(dim)

    # Find paths using different algorithms
    path_bfs = bfs(maze)
    path_dfs = dfs(maze)
    path_astar = a_star(maze)

    # Create subplots to display all algorithms side by side
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    # Animate each algorithm in its subplot
    ani_bfs = animate_path(maze, path_bfs, ax1, color='blue')
    ax1.set_title('BFS')

    ani_dfs = animate_path(maze, path_dfs, ax2, color='green')
    ax2.set_title('DFS')

    ani_astar = animate_path(maze, path_astar, ax3, color='red')
    ax3.set_title('A*')

    # Show the plot
    plt.tight_layout()
    plt.show()
