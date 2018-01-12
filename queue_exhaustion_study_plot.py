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

df = df.sort_values(['batch_interval'])
# df = df.sort_values(['train_enqueue_threads'])

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
                  ymax2):
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

    ax1.bar(time, data2, color=c2, width=50)

    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin2, ymax2)

    # ax2.plot(time, data1, color=c1, alpha=0.75)

    ax2.set_xlim(xmin, xmax)
    ax2.set_ylim(ymin1, ymax1)

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


fig = plt.figure()

plot_num = 0

row_content = df.train_enqueue_threads
row_levels = row_content.unique()

col_content = df.batch_interval
col_levels = col_content.unique()

intraplot_content = df.train_batch_size
intraplot_levels = intraplot_content.unique()


for i, row_level in enumerate(row_levels):

    for j, col_level in enumerate(col_levels):

        plot_num += 1

        # Create scatter axis here.

        ax = fig.add_subplot(len(row_levels),
                             len(col_levels),
                             plot_num)

        ax.set_rasterization_zorder(1)

        show_xlabel = len(row_levels) == (i + 1)

        show_ylabel = j == 0

        annotate_col = i == 0
        col_annotation = r'$ Batch Interval = ' + str(col_level) + '$'

        annotate_row = j == 0
        row_annotation = r'$ Enqueue Threads =' + str(row_level) + ' $ '

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

        # val_loss_mean = run_df.groupby(['step_num'])['val_loss'].mean().tolist()
        # val_loss_std = run_df.groupby(['step_num'])['val_loss'].std().tolist()

        # ax.set_yscale("log", nonposx='clip')

        # ax.loglog()

        ax.set_ylim(0.0, 10000)
        # ax.set_xlim(1, 1000)

        # if plot_error:

        # train_loss_mean = run_df.groupby(['step_num'])['train_error'].mean().tolist()
        # train_loss_std = run_df.groupby(['step_num'])['train_error'].std().tolist()

        # val_loss_mean = run_df.groupby(['step_num'])['val_error'].mean().tolist()
        # val_loss_std = run_df.groupby(['step_num'])['val_error'].std().tolist()

        # ax.set_ylim(0, 1)

        # print(len(val_loss_mean))

        step = run_df['step_num']
        step = run_df.groupby(['step_num'])['step_num'].mean().tolist()

        # step = [mean_running_time * s for s in step]

        bar_line_plot(ax1=ax,
                      time=step,
                      data1=mrt_mean,
                      data1bar=mrt_std,
                      data2=queue_size_mean,
                      c1='r',
                      c2='b',
                      xmin=0,
                      ymin1=0,
                      ymin2=0,
                      xmax=5000,
                      ymax1=0.1,
                      ymax2=10000)

        # line, = ax.plot(step,
        #                 mrt_mean,
        #                 alpha=0.5,
        #                 zorder=0)

        # errorfill(step,
        #           mrt_mean,
        #           mrt_std,
        #           color=line.get_color(),
        #           alpha_fill=0.3, ax=ax)

        # line, = ax.plot(step,
        #                 queue_size_mean,
        #                 "--",
        #                 label=r'Training Loss ($ \mu $)',
        #                 alpha=0.5,
        #                 zorder=0)

        # errorfill(step,
        #           queue_size_mean,
        #           queue_size_std,
        #           color=line.get_color(),
        #           alpha_fill=0.3, ax=ax)

        plt.legend()

        if show_xlabel:

            ax.set_xlabel(r'Simulation Step')

        else:

            ax.xaxis.set_ticklabels([])

        if show_ylabel:

            ax.set_ylabel('Loss')

        else:

            ax.yaxis.set_ticklabels([])

        if annotate_col:
            pad = 5
            ax.annotate(col_annotation, xy=(0.5, 1), xytext=(0, pad),
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')

        if annotate_row:
            pad = -25
            ax.annotate(row_annotation, xy=(0, 0.6), xytext=(pad, 0),
                        rotation=90,
                        xycoords='axes fraction', textcoords='offset points',
                        size='large', ha='center', va='baseline')


plt.grid(True,
         zorder=0)

# plt.legend(bbox_to_anchor=(0.5, 0.0),
#            loc="lower left",
#            mode="expand",
#            bbox_transform=fig.transFigure,
#            borderaxespad=0,
#            ncol=3)

plt.tight_layout(rect=(0.05, 0.05, 0.95, 0.925))

fig.set_size_inches(7.1, 3.5)

plt.suptitle("Comparative Results oif Syanptic Annealing and Stochastic Gradient Descent Optimization")
fig.savefig('C:\\Users\\Justi\\Research\\61\\synaptic_annealing\\figures\\deep_sa_comparative_study.eps',
            rasterized=True,
            dpi=600,
            bbox_inches='tight',
            pad_inches=0.05)
plt.show()
