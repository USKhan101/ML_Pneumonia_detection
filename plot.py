import numpy as np
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt

# Points along the x axis
x = np.linspace(-3.5, 3.5, 1000)
# Normal distribution values
y = norm.pdf(x, 0, 1)  # mean = 0, std deviation = 1

plt.figure(figsize=(12, 6))

# Plot the normal distribution
plt.plot(x, y, 'k-', linewidth=2, label='Standard Normal Distribution')

# different standard deviations
colors = ['blue', 'green', 'red', 'purple']
deviations = [1, 2, 3]  # Standard deviations
labels = ['68% - 1 Std Dev', '95% - 2 Std Dev', '99.7% - 3 Std Dev']

for i, dev in enumerate(deviations):
    plt.fill_between(x, y, where=(x > -dev) & (x < dev), color=colors[i], alpha=0.5, label=labels[i])

plt.title('Standard Normal Distribution with Standard Deviations')
plt.xlabel('Z-score')
plt.ylabel('Probability Density')
plt.legend(loc='upper left')

plt.grid(True)
plt.savefig('./plots/z_score_report.pdf', dpi=300, bbox_inches='tight')
plt.show()

