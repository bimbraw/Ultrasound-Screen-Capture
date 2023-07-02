import numpy as np
import matplotlib.pyplot as plt
import addcopyfighandler

# Data - 3 subjects - Layal removed
c1 = np.array([70, 90, 51, 89])#52, 86])#([95, 69, 94, 50, 74])
c2 = np.array([32, 22, 19, 20])#92, 63])#([78, 85, 92, 66, 73])
c3 = np.array([36, 44, 37, 54])#68, 62])#([64, 41, 50, 66, 73])

# Calculate the average
c1_mean = np.mean(c1)
c2_mean = np.mean(c2)
c3_mean = np.mean(c3)

# Calculate the standard deviation
c1_std = np.std(c1)
c2_std = np.std(c2)
c3_std = np.std(c3)

print(c1_mean,
      c2_mean,
      c3_mean)

print(c1_std,
      c2_std,
      c3_std)

# Define labels, positions, bar heights and error bar heights
labels = ['Subject 1', 'Subject 2', 'Subject 3']
x_pos = np.arange(len(labels))
CTEs = [c1_mean,
        c2_mean,
        c3_mean]

error =[c1_std,
        c2_std,
        c3_std]

#font = {'size'   : 14}

#plt.rc('font', **font)

# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs,
       yerr=error,
       align='center',
       alpha=0.5,
       ecolor='black',
       capsize=5)
ax.set_ylabel('Accuracy Values')
ax.set_xlabel('Subject IDs')
ax.set_xticks(x_pos)
ax.set_xticklabels(labels)
ax.set_ylim([0, 100])
ax.set_title('Accuracy Plot for subjects (averaged over kernels)')
ax.yaxis.grid(True)

# Save the figure and show
plt.tight_layout()
#plt.savefig('bar_plot_with_error_bars_configs_reviewer_6.png')
plt.show()