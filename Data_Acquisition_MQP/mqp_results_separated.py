import numpy as np
import matplotlib.pyplot as plt

# Data
c1 = np.array([95, 78, 64, 94])#([95, 69, 94, 50, 74])
c2 = np.array([69, 85, 41, 79])#([78, 85, 92, 66, 73])
c3 = np.array([94, 92, 50, 84])#([64, 41, 50, 66, 73])
c4 = np.array([50, 66, 66, 63])#([94, 79, 84, 63, 62])
c5 = np.array([74, 73, 73, 62])#([])

# Calculate the average
c1_mean = np.mean(c1)
c2_mean = np.mean(c2)
c3_mean = np.mean(c3)
c4_mean = np.mean(c4)
c5_mean = np.mean(c5)

# Calculate the standard deviation
c1_std = np.std(c1)
c2_std = np.std(c2)
c3_std = np.std(c3)
c4_std = np.std(c4)
c5_std = np.std(c5)

print(c1_mean,
      c2_mean,
      c3_mean,
      c4_mean,
      c5_mean)

print(c1_std,
      c2_std,
      c3_std,
      c4_std,
      c5_std)

# Define labels, positions, bar heights and error bar heights
labels = ['P1', 'P2', 'P3', 'P_U', 'P_D']
x_pos = np.arange(len(labels))
CTEs = [c1_mean,
        c2_mean,
        c3_mean,
        c4_mean,
        c5_mean]
error =[c1_std,
        c2_std,
        c3_std,
        c4_std,
        c5_std]

font = {'size'   : 14}

plt.rc('font', **font)

# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs,
       yerr=error,
       align='center',
       alpha=0.5,
       ecolor='black',
       capsize=10)
ax.set_ylabel('Accuracy Values')
ax.set_xlabel('Probe Configurations')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylim([0, 100])
ax.set_title('Accuracy Plot (Perpendicular Separated)')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
#plt.savefig('bar_plot_with_error_bars_configs_reviewer_6.png')
plt.show()