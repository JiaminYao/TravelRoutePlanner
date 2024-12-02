import matplotlib.pyplot as plt
import numpy as np

# Data
cities = [18, 15, 12, 9, 6]

# Min Cost Path Data
greedy_regular_cost = [1263.76, 1092.76, 998.15, 765.32, 644.77]
greedy_highway_cost = [1518.77, 1222.31, 1223.9, 883.11, 692.8]
dc_regular_cost = [2433.1, 1773.38, 1433.21, 1168.74, 644.77]
dc_highway_cost = [2720.95, 1927.22, 1552.79, 1213.34, 692.8]
dp_regular_cost = [1137.14, 1062.36, 1038.7, 827.32, 784.67]
dp_highway_cost = [1453.25, 1345.55, 1293.74, 1126.61, 960.21]

# Min Time Path Data
greedy_regular_time = [7840, 6842, 6209, 4818, 5363]
greedy_highway_time = [4469, 3940, 3749, 2716, 2195]
dc_regular_time = [17674, 12687, 10349, 8647, 5363]
dc_highway_time = [8119, 5935, 4865, 3923, 2195]
dp_regular_time = [7797, 6889, 7442, 6263, 6094]
dp_highway_time = [4286, 4042, 3785, 3018, 2650]

# Plot Min Cost Path
plt.figure(figsize=(12, 8))
plt.plot(cities, greedy_regular_cost, marker='o', linestyle='-', label='Greedy Regular (Cost)', color='blue')
plt.plot(cities, greedy_highway_cost, marker='s', linestyle='--', label='Greedy Highway (Cost)', color='lightblue')
plt.plot(cities, dc_regular_cost, marker='o', linestyle='-', label='Divide-and-Conquer Regular (Cost)', color='red')
plt.plot(cities, dc_highway_cost, marker='s', linestyle='--', label='Divide-and-Conquer Highway (Cost)', color='lightcoral')
plt.plot(cities, dp_regular_cost, marker='o', linestyle='-', label='Dynamic Programming Regular (Cost)', color='green')
plt.plot(cities, dp_highway_cost, marker='s', linestyle='--', label='Dynamic Programming Highway (Cost)', color='lightgreen')
plt.title('Driving Minimum Cost Comparison of Different Algorithms')
plt.xlabel('Number of Cities')
plt.ylabel('Total Cost ($)')
plt.xticks(cities)
plt.grid(True)
plt.legend()
plt.savefig('./Image/Comparison_Driving_MinCost.png')
plt.show()

# Convert time in minutes to days for the y-axis
max_time = max(
    greedy_regular_time
    + greedy_highway_time
    + dc_regular_time
    + dc_highway_time
    + dp_regular_time
    + dp_highway_time
)
y_ticks_days = np.arange(0, (max_time // (24 * 60)) + 2, 1)  # Generate y-ticks for days

# Plot Min Time Path with days as y-axis
plt.figure(figsize=(12, 8))
plt.plot(cities, np.array(greedy_regular_time) / (24 * 60), marker='o', linestyle='-', label='Greedy Regular (Time)', color='blue')
plt.plot(cities, np.array(greedy_highway_time) / (24 * 60), marker='s', linestyle='--', label='Greedy Highway (Time)', color='lightblue')
plt.plot(cities, np.array(dc_regular_time) / (24 * 60), marker='o', linestyle='-', label='Divide-and-Conquer Regular (Time)', color='red')
plt.plot(cities, np.array(dc_highway_time) / (24 * 60), marker='s', linestyle='--', label='Divide-and-Conquer Highway (Time)', color='lightcoral')
plt.plot(cities, np.array(dp_regular_time) / (24 * 60), marker='o', linestyle='-', label='Dynamic Programming Regular (Time)', color='green')
plt.plot(cities, np.array(dp_highway_time) / (24 * 60), marker='s', linestyle='--', label='Dynamic Programming Highway (Time)', color='lightgreen')

plt.title('Driving Minimum Time Comparison of Different Algorithms')
plt.xlabel('Number of Cities')
plt.ylabel('Total Time (days)')
plt.xticks(cities)
plt.grid(True)
plt.legend()
plt.savefig('./Image/Comparison_Driving_MinTime.png')
plt.show()
