import numpy as np
import matplotlib.pyplot as plt
import time

# Define the Lorenz 86 function model

def lorenz96(t, v):
    dXdt = np.zeros_like(v)
    # iterate over all the indeces
    for i in range(N):
        dXdt[i] =(v[(i+1)%N] - v[i-2])*v[i-1] - v[i] + F
    return dXdt
        
def rk4_step(t, y, h):
    k1 = h * lorenz96(t, y)
    k2 = h * lorenz96(t + h/2, y + k1/2)
    k3 = h * lorenz96(t + h/2, y + k2/2)
    k4 = h * lorenz96(t +h, y +k3)

    return y + (k1 + 2*k2 + 2*k3 + k4)/6


# set up parameters
N = 8 
F = 8  

t0=0
tn=5000 
h=0.005 

time_steps = np.arange(t0, tn, h)

# initialize the variables

v = np.zeros([len(time_steps), N], float)
v[0, :] = F  # Set all values to F
v[0, 0] += 0.5  # Small perturbation to first element

#  Run simulation

start_time = time.time()
for i in range(len(time_steps) - 1):
    v[i + 1] = rk4_step(time_steps[i], v[i], h)
end_time = time.time()

# Compute elapsed time
elapsed_time = end_time - start_time
print(f"Simulation time: {elapsed_time:.4f} seconds")
print(f"Shape of v: {v.shape}")
# Create subplots (1 row, 2 columns)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# First plot: Heatmap of state evolution
im1 = axes[0].imshow(v.T, aspect='auto', cmap='viridis')
fig.colorbar(im1, ax=axes[0], label='State value')
axes[0].set_xlabel('Time step')
axes[0].set_ylabel('Variable index')
axes[0].set_title('Lorenz 96 Model Simulation')

# Second plot: Time series of a single variable (e.g., first variable)
axes[1].plot(time_steps, v[:, 0], label='Variable 1', color='b')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('State value')
axes[1].set_title('Time Series of Variable 1')
axes[1].legend()

# Adjust layout for better spacing
plt.tight_layout()

# Show the plots
plt.show()
