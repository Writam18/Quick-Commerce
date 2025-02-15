import cuopt
import numpy as np
import requests
import networkx as nx
import googlemaps

# Step 1: Load Data (Delivery Locations & Traffic Data)
order_locations = np.array([
    [79.08, 21.14], [79.10, 21.14], [79.06, 21.13],
    [79.09, 21.12], [79.07, 21.15]
])  # Order latitudes & longitudes

dark_store_location = np.array([[79.085, 21.135]])  # Store location

# Step 2: Fetch Real-Time Traffic Data from Google Maps API
GOOGLE_MAPS_API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"
gmaps = googlemaps.Client(key=GOOGLE_MAPS_API_KEY)

def get_travel_time(origin, destination):
    directions = gmaps.directions(origin, destination, mode="driving", departure_time="now")
    return directions[0]['legs'][0]['duration_in_traffic']['value'] / 60  # Convert to minutes

#Compute Traffic-Based Distance Matrix
num_locations = len(order_locations)
distance_matrix = np.zeros((num_locations + 1, num_locations + 1))  # Include dark store

for i in range(num_locations + 1):
    for j in range(num_locations + 1):
        if i != j:
            origin = f"{dark_store_location[0][1]},{dark_store_location[0][0]}" 
            if i == 0 else f"{order_locations[i-1][1]},{order_locations[i-1][0]}"
            destination = f"{order_locations[j-1][1]},{order_locations[j-1][0]}"
            distance_matrix[i][j] = get_travel_time(origin, destination)

# Solve Vehicle Routing Problem Using NVIDIA cuOpt
problem = cuopt.VehicleRoutingProblem(num_vehicles=1, num_orders=num_locations)
problem.set_cost_matrix(distance_matrix)

# Set order time constraints (15-20 mins)
for i in range(num_locations):
    problem.set_time_window(i + 1, min_time=0, max_time=20)

# Solve the optimization problem
solver = cuopt.Solver(problem)
solution = solver.solve()

# Get Optimized Route & Display
optimal_route = solution.get_route(0)  

print("Optimized Delivery Route:", optimal_route)

# Step 6: Visualize Optimized Route
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(10, 8))

# Plot dark store
ax.scatter(dark_store_location[:, 0], dark_store_location[:, 1], c='red', marker='s', label="Dark Store", s=200)

# Plot order locations
ax.scatter(order_locations[:, 0], order_locations[:, 1], c='blue', label="Orders", s=100)

# Plot optimized route
for i in range(len(optimal_route) - 1):
    x1, y1 = dark_store_location[0] if optimal_route[i] == 0 else order_locations[optimal_route[i] - 1]
    x2, y2 = dark_store_location[0] if optimal_route[i+1] == 0 else order_locations[optimal_route[i+1] - 1]
    ax.plot([x1, x2], [y1, y2], 'k--', linewidth=1.5)

ax.legend()
ax.set_title("Optimized AI-Based Delivery Route (15-20 min constraint)")
plt.show()
