import cv2
import numpy as np
import math
import heapq
import matplotlib.pyplot as plt
from matplotlib import cm

# Load map.pgm (replace with your path if needed)
map_img = cv2.imread('../maps/map.pgm', cv2.IMREAD_GRAYSCALE)
if map_img is None:
    raise FileNotFoundError("map.pgm could not be loaded")

map_img = cv2.flip(map_img, 0)  # Flip to match ROS orientation

# Thresholds from map.yaml
occupied_thresh = 0.9
free_thresh = 0.25

# Convert to occupancy grid
grid = np.full_like(map_img, -1, dtype=np.int8)
grid[map_img >= int(free_thresh * 255)] = 1
grid[map_img <= int(occupied_thresh * 255)] = 0

# Sample positions to visit (mock camera positions in pixel space)
positions = [(30, 30), (40, 40), (80, 120), (35, 90), (35, 110)]

# A* functions
def heuristic(a, b):
    return math.hypot(b[0] - a[0], b[1] - a[1])

def astar(start, goal, grid):
    height, width = grid.shape
    visited = set()
    queue = [(heuristic(start, goal), 0, start)]
    g_score = {start: 0}

    while queue:
        _, cost, current = heapq.heappop(queue)
        if current == goal:
            return cost
        if current in visited:
            continue
        visited.add(current)

        for dx, dy in [(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1)]:
            neighbor = (current[0] + dx, current[1] + dy)
            if not (0 <= neighbor[0] < width and 0 <= neighbor[1] < height):
                continue
            if grid[neighbor[1], neighbor[0]] != 1:
                continue

            tentative_g = g_score[current] + math.hypot(dx, dy)
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                g_score[neighbor] = tentative_g
                priority = tentative_g + heuristic(neighbor, goal)
                heapq.heappush(queue, (priority, tentative_g, neighbor))

    return float('inf')

# Compute distances between all pairs and find the shortest visiting order (greedy for now)
visited = [0]
unvisited = set(range(1, len(positions)))
sequence = [0]

while unvisited:
    last = visited[-1]
    next_idx = min(unvisited, key=lambda i: astar(positions[last], positions[i], grid))
    sequence.append(next_idx)
    visited.append(next_idx)
    unvisited.remove(next_idx)

# Draw
map_color = cv2.cvtColor(map_img, cv2.COLOR_GRAY2BGR)
for idx, (x, y) in enumerate(positions):
    cv2.circle(map_color, (x, y), 5, (0, 0, 255), -1)
    cv2.putText(map_color, str(idx), (x+6, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# Draw paths
for i in range(len(sequence) - 1):
    pt1 = positions[sequence[i]]
    pt2 = positions[sequence[i+1]]
    cv2.line(map_color, pt1, pt2, (255, 0, 0), 1)

# Convert to RGB for matplotlib
map_rgb = cv2.cvtColor(map_color, cv2.COLOR_BGR2RGB)

# Show image
plt.figure(figsize=(10, 10))
plt.imshow(map_rgb)
plt.title("A* Test Path on Map")
plt.axis('off')
plt.show()

