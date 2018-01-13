from __future__ import print_function

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams.update({'font.size': 6})

plt.style.use('seaborn-whitegrid')

# df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/queue_exhaustion_study.csv')
df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/tensorflow_experiment_merged.csv')

df = df.loc[(df.step_num != 0)]

df = df.sort_values(['batch_interval', 'train_enqueue_threads'])

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

df_bi_grouped = df.groupby(['batch_interval'])
mean_dequeue_rate_mean = df_bi_grouped['mean_dequeue_rate'].mean().tolist()
mean_enqueue_rate_mean = df_bi_grouped['mean_enqueue_rate'].mean().tolist()
batch_intervals = df_bi_grouped['batch_interval'].mean().tolist()

ax.scatter(batch_intervals,
           mean_dequeue_rate_mean,
           color='r',
           alpha=0.3)

ax.scatter(batch_intervals,
           mean_enqueue_rate_mean,
           color='b',
           alpha=0.3)


plt.suptitle("Comparative Results oif Syanptic Annealing and Stochastic Gradient Descent Optimization")
# fig.savefig('C:\\Users\\Justi\\Research\\61\\synaptic_annealing\\figures\\deep_sa_comparative_study.eps',
#             rasterized=True,
#             dpi=600,
#             bbox_inches='tight',
#             pad_inches=0.05)
plt.show()

# df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/queue_exhaustion_study.csv')
df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/tensorflow_experiment_merged.csv')

df = df.loc[(df.step_num != 0)]

# df = df.sort_values(['batch_interval', 'train_enqueue_threads'])

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)

df_enqueue_thread_grouped = df.groupby(['train_enqueue_threads'])
mean_dequeue_rate_mean = df_enqueue_thread_grouped['mean_dequeue_rate'].mean().tolist()
mean_enqueue_rate_mean = df_enqueue_thread_grouped['mean_enqueue_rate'].mean().tolist()
train_enqueue_threads = df_enqueue_thread_grouped['train_enqueue_threads'].mean().tolist()

ax.scatter(train_enqueue_threads,
           mean_dequeue_rate_mean,
           color='r',
           alpha=0.3)

ax.scatter(train_enqueue_threads,
           mean_enqueue_rate_mean,
           color='b',
           alpha=0.3)

plt.suptitle("Comparative Results oif Syanptic Annealing and Stochastic Gradient Descent Optimization")
# fig.savefig('C:\\Users\\Justi\\Research\\61\\synaptic_annealing\\figures\\deep_sa_comparative_study.eps',
#             rasterized=True,
#             dpi=600,
#             bbox_inches='tight',
#             pad_inches=0.05)
plt.show()
