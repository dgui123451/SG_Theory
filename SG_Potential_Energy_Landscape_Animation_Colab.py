import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import math

# --- Define the Potential Energy Function V(phi_plus, phi_minus) ---
# V = (1/2)m²(φ+)² - (1/2)m²(φ-)² + (λ/24)[(φ+)⁴+(φ-)⁴] + (3m⁴)/(2λ)
def potential_V(phi_plus, phi_minus, m, lambda_coupling):
    if lambda_coupling == 0:
        # Avoid division by zero, return a large number or handle error
        return float('inf')

    term1 = 0.5 * m**2 * phi_plus**2
    term2 = -0.5 * m**2 * phi_minus**2
    term3 = (lambda_coupling / 24.0) * (phi_plus**4 + phi_minus**4)
    term4 = (3 * m**4) / (2 * lambda_coupling)

    total_V = term1 + term2 + term3 + term4
    return total_V, term1, term2, term3, term4

# --- Define the Gradients of V (for "rolling downhill") ---
# ∂V/∂φ+ = m²φ+ + (λ/6)φ+³
def grad_V_phi_plus(phi_plus, m, lambda_coupling):
    return m**2 * phi_plus + (lambda_coupling / 6.0) * phi_plus**3

# ∂V/∂φ- = -m²φ- + (λ/6)φ-³
def grad_V_phi_minus(phi_minus, m, lambda_coupling):
    return -m**2 * phi_minus + (lambda_coupling / 6.0) * phi_minus**3

# --- Simulation and Animation Parameters ---
m_param = 1.0
lambda_param = 1.0 # Ensure lambda_param is not zero

# Animation settings
n_frames = 200  # Number of frames in the animation
dt_animation = 0.05  # Time step for the particle's movement (controls speed)
learning_rate = 0.1 # How big a step the particle takes downhill

# Plotting range for phi_plus and phi_minus
phi_range = 3.5
phi_plus_vals = np.linspace(-phi_range, phi_range, 50)
phi_minus_vals = np.linspace(-phi_range, phi_range, 50)
Phi_plus_grid, Phi_minus_grid = np.meshgrid(phi_plus_vals, phi_minus_vals)

# Calculate V over the grid
V_grid, _, _, _, _ = potential_V(Phi_plus_grid, Phi_minus_grid, m_param, lambda_param)

# Initial position of the "particle" (state of the fields)
# Let's start it somewhere away from the minimum
phi_plus_current = 2.0
phi_minus_current = 1.0

# Store the path of the particle
path_phi_plus = [phi_plus_current]
path_phi_minus = [phi_minus_current]
path_V = [potential_V(phi_plus_current, phi_minus_current, m_param, lambda_param)[0]]


# --- Set up the 3D Plot ---
fig = plt.figure(figsize=(12, 9))
ax = fig.add_subplot(111, projection='3d')

# Plot the potential energy surface
surf = ax.plot_surface(Phi_plus_grid, Phi_minus_grid, V_grid, cmap='viridis', alpha=0.7, edgecolor='none')

# Line for the particle's path and a point for its current position
line, = ax.plot([], [], [], 'r-', lw=2, label='Path of Fields (φ+, φ-)') # Path
point, = ax.plot([], [], [], 'ro', markersize=8, label='Current State (φ+, φ-)') # Current position

ax.set_xlabel('Positive Field (φ+)', fontsize=12)
ax.set_ylabel('Negative Field (φ-)', fontsize=12)
ax.set_zlabel('Potential Energy V(φ+, φ-)', fontsize=12)
ax.set_title('Symmetrodynamic Gravity: Potential Energy Landscape', fontsize=14)
ax.legend(loc='upper left')
fig.colorbar(surf, shrink=0.5, aspect=5, label='Energy V')

# Text annotations for dynamic values
time_text = ax.text2D(0.02, 0.95, '', transform=ax.transAxes, color='black', fontsize=10,
                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='grey'))
values_text = ax.text2D(0.02, 0.05, '', transform=ax.transAxes, color='black', fontsize=9,
                        bbox=dict(facecolor='white', alpha=0.7, edgecolor='grey'),
                        verticalalignment='bottom')


# --- Animation Update Function ---
def update(frame):
    global phi_plus_current, phi_minus_current

    # Calculate gradients at the current position
    grad_p = grad_V_phi_plus(phi_plus_current, m_param, lambda_param)
    grad_m = grad_V_phi_minus(phi_minus_current, m_param, lambda_param)

    # Update positions by moving in the direction of the negative gradient (downhill)
    phi_plus_current -= learning_rate * grad_p * dt_animation
    phi_minus_current -= learning_rate * grad_m * dt_animation

    # Store path
    path_phi_plus.append(phi_plus_current)
    path_phi_minus.append(phi_minus_current)

    current_V, t1, t2, t3, t4 = potential_V(phi_plus_current, phi_minus_current, m_param, lambda_param)
    path_V.append(current_V)

    # Update plot elements
    line.set_data(path_phi_plus, path_phi_minus)
    line.set_3d_properties(path_V)

    point.set_data([phi_plus_current], [phi_minus_current])
    point.set_3d_properties([current_V])

    # Update text annotations
    time_text.set_text(f'Frame: {frame+1}/{n_frames}')

    values_str = (
        f"Current State:\n"
        f"  φ+ = {phi_plus_current:.3f}\n"
        f"  φ- = {phi_minus_current:.3f}\n"
        f"  V_total = {current_V:.3f}\n\n"
        f"Term Contributions:\n"
        f"  T1 (φ+²): {t1:.3f}\n"
        f"  T2 (φ-²): {t2:.3f}\n"
        f"  T3 (φ⁴):  {t3:.3f}\n"
        f"  T4 (Const): {t4:.3f}"
    )
    values_text.set_text(values_str)

    # Adjust view for better visualization if needed (optional)
    # ax.view_init(elev=30., azim=frame*0.5)

    return line, point, time_text, values_text

# Create the animation
# Important: Close the initial static plot for Colab animation
plt.close(fig)

print("Creating animation (this might take a moment)...")
ani = FuncAnimation(fig, update, frames=n_frames, interval=100, blit=False) # blit=False for 3D usually

# Convert to HTML5 video for Colab
html_video = ani.to_html5_video()
print("Displaying animation...")

display(HTML(html_video))
print("Animation complete.")

# Print the predicted vacuum values for comparison
vac_phi_minus_sq = (6 * m_param**2) / lambda_param
vac_phi_minus = math.sqrt(vac_phi_minus_sq) if vac_phi_minus_sq >= 0 else float('nan')
print(f"\nFor m={m_param}, lambda={lambda_param}:")
print(f"  Predicted vacuum: φ+ = 0, φ- = ±{vac_phi_minus:.3f}")
print(f"  Predicted vacuum energy V = 0")
