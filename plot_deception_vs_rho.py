import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import matplotlib.ticker as ticker

# Set up publication-quality figure settings
plt.style.use('seaborn-v0_8-whitegrid')
rcParams['font.family'] = 'serif'
# Disable LaTeX rendering
rcParams['text.usetex'] = False
rcParams['figure.figsize'] = (8, 6)
rcParams['axes.labelsize'] = 14
rcParams['xtick.labelsize'] = 12
rcParams['ytick.labelsize'] = 12
rcParams['legend.fontsize'] = 12
rcParams['figure.titlesize'] = 16

# Function to calculate r_deception based on the formula
def r_deception(rho):
    return 0.25 - np.arcsin(rho) / (2 * np.pi)

# Create data points
rho_values = np.linspace(-1, 1, 1000)
r_values = r_deception(rho_values)

# Create the figure
fig, ax = plt.subplots()

# Plot the main curve
ax.plot(rho_values, r_values, 'b-', linewidth=2.5)

# Plot the special case where rho=0 and r_deception=0.25
ax.plot(0, 0.25, 'ro', markersize=8)

# Add horizontal and vertical lines at special points
ax.axhline(y=0.25, color='gray', linestyle='--', alpha=0.5)
ax.axvline(x=0, color='gray', linestyle='--', alpha=0.5)

# Add more reference points (without annotations)
ax.plot(-1, 0.5, 'go', markersize=6)
ax.plot(1, 0, 'go', markersize=6)

# Set labels and title
ax.set_xlabel('Correlation coefficient (ρ)')
ax.set_ylabel('Deception ratio (r_deception)')
ax.set_title('Relationship between ρ and r_deception')

# Beautify the figure
ax.grid(True, alpha=0.3)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Set tick marks
ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.05))

# Add formula as text (simplified)
ax.text(0.05, 0.95, 'r_deception = 1/4 - arcsin(ρ)/2π',
        transform=ax.transAxes, fontsize=12, va='top',
        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))

# Add x and y axis limits
ax.set_xlim([-1.05, 1.05])
ax.set_ylim([-0.05, 0.55])

# Save the figure
plt.tight_layout()
plt.savefig('figures/deception_vs_rho.pdf', bbox_inches='tight', dpi=300)
plt.savefig('figures/deception_vs_rho.png', bbox_inches='tight', dpi=300)

# Show the figure
plt.show() 