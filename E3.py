import matplotlib.pyplot as plt

# Data
cities = [18, 15, 12, 9, 6]

# Min Cost Path Data
greedy_cost = [1464, 1269, 794, 531, 343]
dc_cost = [2159, 1750, 1189, 703, 487]
dp_cost = [1331, 1062, 782, 508, 333]

# Min Time Path Data
greedy_time = [1670, 1130, 980, 750, 520]
dc_time = [2030, 1290, 1040, 815, 490]
dp_time = [1670, 1290, 990, 765, 570]

# Plot Min Cost Path
plt.figure(figsize=(12, 8))
plt.plot(cities, greedy_cost, marker='o', label='Greedy Min Cost', color='blue')
plt.plot(cities, dc_cost, marker='o', label='Divide-and-Conquer Min Cost', color='red')
plt.plot(cities, dp_cost, marker='o', label='Dynamic Programming Min Cost', color='green')
plt.title('Flight Minimum Cost Path Comparison of Different Algorithms')
plt.xlabel('Number of Cities')
plt.ylabel('Total Cost ($)')
plt.xticks(cities)
plt.grid(True)
plt.legend()
plt.savefig('./Image/Comparison_Flight_MinCost.png')
plt.show()

# Plot Min Time Path
plt.figure(figsize=(12, 8))
plt.plot(cities, greedy_time, marker='o', label='Greedy Min Time', color='blue')
plt.plot(cities, dc_time, marker='o', label='Divide-and-Conquer Min Time', color='red')
plt.plot(cities, dp_time, marker='o', label='Dynamic Programming Min Time', color='green')
plt.title('Flight Minimum Time Path Comparison of Different Algorithms')
plt.xlabel('Number of Cities')
plt.ylabel('Total Time (mins)')
plt.xticks(cities)
plt.grid(True)
plt.legend()
plt.savefig('./Image/Comparison_Flight_MinTime.png')
plt.show()
