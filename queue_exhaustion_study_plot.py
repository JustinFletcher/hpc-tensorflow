from __future__ import print_function

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from matplotlib import cm


# df = df.drop(df[df.optimizer == 'sgd'].index)


def bar_line_plot(ax1,
                  time,
                  data1,
                  data1bar,
                  data2,
                  c1,
                  c2,
                  xmin,
                  ymin1,
                  ymin2,
                  xmax,
                  ymax1,
                  ymax2,
                  show_xlabel,
                  show_ylabel_1,
                  show_ylabel_2,
                  annotate_col,
                  col_annotation,
                  annotate_row,
                  row_annotation):
    """
    Parameters
    ----------
    ax : axis
        Axis to put two scales on

    time : array-like
        x-axis values for both datasets

    data1: array-like
        Data for left hand scale

    data2 : array-like
        Data for right hand scale

    c1 : color
        Color for line 1

    c2 : color
        Color for line 2

    Returns
    -------
    ax : axis
        Original axis
    ax2 : axis
        New twin axis
    """
    ax2 = ax1.twinx()

    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin2, ymax2)

    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin1, ymax1)

    ax1.fill_between(time, 0, data2, color=c2, alpha=0.3)

    if show_xlabel:
        ax1.set_xlabel('Training Step')
    else:
        ax1.xaxis.set_ticklabels([])

    if show_ylabel_1:
        ax1.set_ylabel('Queue Size')
    else:
        ax1.yaxis.set_ticklabels([])

    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin2, ymax2)

    line, = ax2.plot(step,
                     data1,
                     color=c1,
                     alpha=0.5,
                     zorder=0)

    errorfill(step,
              data1,
              data1bar,
              color=line.get_color(),
              alpha_fill=0.3,
              ax=ax2)

    if show_ylabel_2:
        ax2.set_ylabel('Mean Single \n Batch Inference \n Running Time')
    else:
        ax2.yaxis.set_ticklabels([])

    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin1, ymax1)

    if annotate_col:
        pad = 10
        ax1.annotate(col_annotation, xy=(0.5, 1), xytext=(0, pad),
                     xycoords='axes fraction', textcoords='offset points',
                     size='large', ha='center', va='baseline')

    if annotate_row:
        pad = -70
        ax1.annotate(row_annotation, xy=(0, 0.75), xytext=(pad, 0),
                     rotation=90,
                     xycoords='axes fraction', textcoords='offset points',
                     size='large', ha='center', va='baseline')

    return ax1, ax2


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, ax=None):

    ax = ax if ax is not None else plt.gca()

    if np.isscalar(yerr) or len(yerr) == len(y):

        ymin = [y_i - yerr_i for (y_i, yerr_i) in zip(y, yerr)]
        ymax = [y_i + yerr_i for (y_i, yerr_i) in zip(y, yerr)]

    elif len(yerr) == 2:

        ymin, ymax = yerr

    # ax.plot(x, y)
    ax.fill_between(x, ymax, ymin, alpha=alpha_fill, color=color,
                    zorder=0)


# -------------------------------------------------


matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams.update({'font.size': 6})

plt.style.use('seaborn-whitegrid')

df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/queue_exhaustion_study.csv')
# df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/tensorflow_experiment_merged.csv')

df = df.sort_values(['batch_interval', 'train_enqueue_threads'])

df = df.loc[(df.step_num > 30)]
df = df.loc[(df.pause_time == 10)]

df = df.loc[(df.queue_size > 100)]
df = df.loc[(df.queue_size < 99500)]

row_content = df.train_enqueue_threads
row_levels = row_content.unique()

col_content = df.batch_interval
col_levels = col_content.unique()

intraplot_content = df.train_batch_size
intraplot_levels = intraplot_content.unique()


enqueue_rates = []
X = []
Y = []

for i, row_level in enumerate(row_levels):

    for j, col_level in enumerate(col_levels):

        run_df = df.loc[(row_content == row_level) &
                        (col_content == col_level)]

        queue_size_mean = run_df.groupby(['step_num'])['queue_size'].mean().tolist()
        queue_size_std = run_df.groupby(['step_num'])['queue_size'].std().tolist()
        enqueue_rate = (queue_size_mean[-1] - queue_size_mean[0]) / (len(queue_size_mean))
        enqueue_rates.append(enqueue_rate)
        X.append(row_level)
        Y.append(col_level)

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_trisurf(X,
                       Y,
                       enqueue_rates,
                       cmap=cm.coolwarm,
                       linewidth=0.1,
                       antialiased=True)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


# -------------------------------------------------


# -------------------------------------------------


matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams.update({'font.size': 6})

plt.style.use('seaborn-whitegrid')

df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/queue_exhaustion_study.csv')
# df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/tensorflow_experiment_merged.csv')

df = df.sort_values(['batch_interval', 'train_enqueue_threads'])

df = df.loc[(df.step_num > 30)]
df = df.loc[(df.pause_time == 10)]


row_content = df.train_enqueue_threads
row_levels = row_content.unique()

col_content = df.batch_interval
col_levels = col_content.unique()

intraplot_content = df.train_batch_size
intraplot_levels = intraplot_content.unique()


mean_running_times = []
X = []
Y = []

for i, row_level in enumerate(row_levels):

    for j, col_level in enumerate(col_levels):

        run_df = df.loc[(row_content == row_level) &
                        (col_content == col_level)]

        mean_running_time_mean = run_df['mean_running_time'].mean()
        mean_running_time_std = run_df['mean_running_time'].std()
        mean_running_times.append(mean_running_time_mean)
        X.append(row_level)
        Y.append(col_level)

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_trisurf(X,
                       Y,
                       mean_running_times,
                       cmap=cm.coolwarm,
                       linewidth=0.1,
                       antialiased=True)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


# -------------------------------------------------

# -------------------------------------------------


matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams.update({'font.size': 6})

plt.style.use('seaborn-whitegrid')

df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/queue_exhaustion_study.csv')
# df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/tensorflow_experiment_merged.csv')

df = df.sort_values(['batch_interval', 'train_enqueue_threads'])

df = df.loc[(df.step_num > 30)]
df = df.loc[(df.pause_time == 10)]


row_content = df.train_enqueue_threads
row_levels = row_content.unique()

col_content = df.batch_interval
col_levels = col_content.unique()

intraplot_content = df.train_batch_size
intraplot_levels = intraplot_content.unique()


mean_running_times = []
X = []
Y = []

for i, row_level in enumerate(row_levels):

    for j, col_level in enumerate(col_levels):

        run_df = df.loc[(row_content == row_level) &
                        (col_content == col_level)]

        val_loss_mean = run_df.groupby(['step_num'])['val_error'].mean().tolist()
        val_loss_std = run_df.groupby(['step_num'])['val_loss'].std().tolist()
        mean_running_times.append(val_loss_mean[-1])
        X.append(row_level)
        Y.append(col_level)

fig = plt.figure()
ax = fig.gca(projection='3d')

surf = ax.plot_trisurf(X,
                       Y,
                       mean_running_times,
                       cmap=cm.coolwarm,
                       linewidth=0.1,
                       antialiased=True)

fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()


# -------------------------------------------------



matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['text.latex.unicode'] = True
matplotlib.rcParams.update({'font.size': 6})

plt.style.use('seaborn-whitegrid')

df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/queue_exhaustion_study.csv')
# df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/tensorflow_experiment_merged.csv')

df = df.sort_values(['batch_interval', 'train_enqueue_threads'])

df = df.loc[(df.step_num > 30)]
df = df.loc[(df.pause_time == 10)]


row_content = df.train_enqueue_threads
row_levels = row_content.unique()

col_content = df.batch_interval
col_levels = col_content.unique()

intraplot_content = df.train_batch_size
intraplot_levels = intraplot_content.unique()


fig = plt.figure()

plot_num = 0

for i, row_level in enumerate(row_levels):

    for j, col_level in enumerate(col_levels):

        plot_num += 1

        # Create scatter axis here.

        ax = fig.add_subplot(len(row_levels),
                             len(col_levels),
                             plot_num)

        ax.set_rasterization_zorder(1)

        show_xlabel = len(row_levels) == (i + 1)

        show_ylabel_1 = j == 0
        show_ylabel_2 = len(col_levels) == (j + 1)

        annotate_col = i == 0
        col_annotation = r'Batch Interval $ = ' + str(col_level) + '$'

        annotate_row = j == 0
        row_annotation = r'Threads $=' + str(row_level) + ' $ '

        # ax.set_xlim(0.00001, 10)
        # ax.set_ylim(0.001, 1)

        # for k, intraplot_level in enumerate(intraplot_levels):

        run_df = df.loc[(row_content == row_level) &
                        (col_content == col_level)]

        # run_df = df.loc[(row_content == row_level) &
        #                 (col_content == col_level) &
        #                 (intraplot_content == intraplot_level)]

        # if plot_loss:

        # mean_running_time = run_df['mean_running_time'].mean()
        # print(mean_running_time)

        mrt_mean = run_df.groupby(['step_num'])['mean_running_time'].mean().tolist()
        mrt_std = run_df.groupby(['step_num'])['mean_running_time'].std().tolist()

        queue_size_mean = run_df.groupby(['step_num'])['queue_size'].mean().tolist()
        queue_size_std = run_df.groupby(['step_num'])['queue_size'].std().tolist()

        mean_dequeue_rate_mean = run_df.groupby(['step_num'])['mean_dequeue_rate'].mean().tolist()
        mean_dequeue_rate_std = run_df.groupby(['step_num'])['mean_dequeue_rate'].std().tolist()

        mean_enqueue_rate_mean = run_df.groupby(['step_num'])['mean_enqueue_rate'].mean().tolist()
        mean_enqueue_rate_std = run_df.groupby(['step_num'])['mean_enqueue_rate'].std().tolist()

        # val_loss_mean = run_df.groupby(['step_num'])['val_loss'].mean().tolist()
        # val_loss_std = run_df.groupby(['step_num'])['val_loss'].std().tolist()

        # ax.set_yscale("log", nonposx='clip')

        # ax.loglog()

        # ax.set_ylim(0.0, 10000)
        # ax.set_xlim(1, 1000)

        step = run_df['step_num']
        step = run_df.groupby(['step_num'])['step_num'].mean().tolist()

        # step = [mean_running_time * s for s in step]

        bar_line_plot(ax1=ax,
                      time=step,
                      data1=mrt_mean,
                      data1bar=mrt_std,
                      # data2=[e/d for (e, d) in zip(mean_enqueue_rate_mean, mean_dequeue_rate_mean)],
                      data2=queue_size_mean,
                      c1='r',
                      c2='b',
                      xmin=0,
                      ymin1=0,
                      ymin2=0,
                      xmax=1000,
                      ymax1=0.1,
                      ymax2=100000,
                      show_xlabel=show_xlabel,
                      show_ylabel_1=show_ylabel_1,
                      show_ylabel_2=show_ylabel_2,
                      annotate_col=annotate_col,
                      col_annotation=col_annotation,
                      annotate_row=annotate_row,
                      row_annotation=row_annotation)

        plt.legend()


plt.grid(True,
         zorder=0)

plt.legend(bbox_to_anchor=(0.5, 0.0),
           loc="lower left",
           mode="expand",
           bbox_transform=fig.transFigure,
           borderaxespad=0,
           ncol=3)

plt.tight_layout(rect=(0.05, 0.05, 0.95, 0.925))

fig.set_size_inches(16, 9)

plt.suptitle("")
# fig.savefig('C:\\Users\\Justi\\Research\\61\\synaptic_annealing\\figures\\deep_sa_comparative_study.eps',
#             rasterized=True,
#             dpi=600,
#             bbox_inches='tight',
#             pad_inches=0.05)
plt.show()
