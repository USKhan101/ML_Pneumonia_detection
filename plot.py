import numpy as np
import seaborn as sns
from scipy.stats import norm
import matplotlib.pyplot as plt

# Points along the x axis
x = np.linspace(-3.5, 3.5, 1000)
# Normal distribution values
y = norm.pdf(x, 0, 1)  # mean = 0, std deviation = 1

plt.figure(figsize=(10, 6))

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
plt.savefig('z_score_report.png', dpi=300, bbox_inches='tight')
plt.show()

# Generate sample data
data = np.random.normal(loc=0, scale=1, size=1000)

plt.figure(figsize=(8, 6))
sns.boxplot(data, whis=1.5, width=0.5)  # 'whis' parameter is the IQR multiplier for outlier steps

# Calculate IQR and bounds for whiskers
Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Add lines for lower and upper bounds
plt.axhline(y=lower_bound, color='r', linestyle='--', label='Lower Bound')
plt.axhline(y=upper_bound, color='g', linestyle='--', label='Upper Bound')
plt.title('Boxplot with IQR Boundaries')
plt.xlabel('Data Points')
plt.legend()
plt.savefig('IQR_report.png', dpi=300, bbox_inches='tight')
plt.show()

