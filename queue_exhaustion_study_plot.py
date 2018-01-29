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
                  xlabel,
                  ylabel_1,
                  ylabel_2,
                  show_xlabel,
                  show_ylabel_1,
                  show_ylabel_2,
                  annotate_col,
                  col_annotation,
                  annotate_row,
                  row_annotation,
                  show_legends=True):
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

    ax1.fill_between(time, 0, data2, color=c2, label=ylabel_1, alpha=0.3)

    if show_xlabel:
        ax1.set_xlabel(xlabel)
    else:
        ax1.xaxis.set_ticklabels([])

    if show_ylabel_1:
        ax1.set_ylabel(ylabel_1)
    else:
        ax1.yaxis.set_ticklabels([])

    ax1.set_xlim(xmin, xmax)
    ax1.set_ylim(ymin2, ymax2)

    line, = ax2.plot(time,
                     data1,
                     color=c1,
                     alpha=0.5,
                     label='Mean ' + ylabel_2,
                     zorder=0)

    errorfill(time,
              data1,
              data1bar,
              color=line.get_color(),
              alpha_fill=0.3,
              label=r'$\pm\sigma$ ' + ylabel_2,
              ax=ax2)

    if show_ylabel_2:
        ax2.set_ylabel(ylabel_2)
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

    if show_legends:
        ax1.legend(loc=1)
        ax2.legend(loc=2)

    return ax1, ax2


def errorfill(x, y, yerr, color=None, alpha_fill=0.3, label=None, ax=None):

    ax = ax if ax is not None else plt.gca()

    if np.isscalar(yerr) or len(yerr) == len(y):

        ymin = [y_i - yerr_i for (y_i, yerr_i) in zip(y, yerr)]
        ymax = [y_i + yerr_i for (y_i, yerr_i) in zip(y, yerr)]

    elif len(yerr) == 2:

        ymin, ymax = yerr

    # ax.plot(x, y)
    ax.fill_between(x,
                    ymax,
                    ymin,
                    alpha=alpha_fill,
                    color=color,
                    label=label,
                    zorder=0)


# -------------------------------------------------
# -------------------------------------------------
# -----------Single Plot Example-------------------
# -------------------------------------------------
# -------------------------------------------------

def single_plot_example():

    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True
    matplotlib.rcParams.update({'font.size': 14})

    plt.style.use('seaborn-whitegrid')

    df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/queue_exhaustion_study.csv')
    df = df.sort_values(['batch_interval', 'train_enqueue_threads'])

    df = df.loc[(df.batch_interval == 1)]
    df = df.loc[(df.train_enqueue_threads == 2)]

    # df = df.loc[(df.rep_num == 0)]

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

            annotate_col = False
            col_annotation = r'Batch Interval $ = ' + str(col_level) + '$'

            annotate_row = False
            row_annotation = r'Threads $=' + str(row_level) + ' $ '

            run_df = df.loc[(row_content == row_level) &
                            (col_content == col_level)]

            mrt_mean = run_df.groupby(['step_num'])['mean_running_time'].mean().tolist()
            mrt_std = run_df.groupby(['step_num'])['mean_running_time'].std().tolist()

            queue_size_mean = run_df.groupby(['step_num'])['queue_size'].mean().tolist()
            queue_size_std = run_df.groupby(['step_num'])['queue_size'].std().tolist()

            step = run_df['step_num']
            step = run_df.groupby(['step_num'])['step_num'].mean().tolist()

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
                          xmax=1000,
                          ymax1=0.1,
                          ymax2=100000,
                          xlabel='Training Step',
                          ylabel_1='Queue Size (images)',
                          ylabel_2='Single-Step Running Time (sec)',
                          show_xlabel=show_xlabel,
                          show_ylabel_1=show_ylabel_1,
                          show_ylabel_2=show_ylabel_2,
                          annotate_col=annotate_col,
                          col_annotation=col_annotation,
                          annotate_row=annotate_row,
                          row_annotation=row_annotation)

            # plt.legend()

    plt.grid(True,
             zorder=0)

    # plt.legend(bbox_to_anchor=(0.5, 0.0),
    #            loc="lower left",
    #            mode="expand",
    #            bbox_transform=fig.transFigure,
    #            borderaxespad=0,
    #            ncol=3)

    # plt.tight_layout(rect=(0.001, 0.001, 1.0, 1.0))

    fig.set_size_inches(16, 9)

    plt.suptitle(r"Running Time ($\mu\pm\sigma$, $n=30$) and Queue Size ($\mu$) vs. Training Step ($I_B = 1$, Threads $= 2$)")
    # fig.savefig('C:\\Users\\Justi\\Research\\61\\synaptic_annealing\\figures\\deep_sa_comparative_study.eps',
    #             rasterized=True,
    #             dpi=600,
    #             bbox_inches='tight',
    #             pad_inches=0.05)
    plt.show()

# -------------------------------------------------
# -------------------------------------------------
# ----------Thread Count Column--------------------
# -------------------------------------------------


def thread_count_col():

    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True
    matplotlib.rcParams.update({'font.size': 10})

    plt.style.use('seaborn-whitegrid')

    df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/queue_exhaustion_study.csv')
    df = df.sort_values(['batch_interval', 'train_enqueue_threads'])

    df = df.loc[(df.batch_interval == 1)]
    df = df.loc[(df.train_enqueue_threads != 128)]

    # df = df.loc[(df.rep_num == 0)]

    row_content = df.train_enqueue_threads
    row_levels = row_content.unique()

    col_content = df.batch_interval
    col_levels = col_content.unique()

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

            run_df = df.loc[(row_content == row_level) &
                            (col_content == col_level)]

            mrt_mean = run_df.groupby(['step_num'])['mean_running_time'].mean().tolist()
            mrt_std = run_df.groupby(['step_num'])['mean_running_time'].std().tolist()

            queue_size_mean = run_df.groupby(['step_num'])['queue_size'].mean().tolist()
            queue_size_std = run_df.groupby(['step_num'])['queue_size'].std().tolist()

            step = run_df['step_num']
            step = run_df.groupby(['step_num'])['step_num'].mean().tolist()

            ax1, ax2 = bar_line_plot(ax1=ax,
                                     time=step,
                                     data1=mrt_mean,
                                     data1bar=mrt_std,
                                     data2=queue_size_mean,
                                     c1='r',
                                     c2='b',
                                     xmin=0,
                                     ymin1=0,
                                     ymin2=0,
                                     xmax=2500,
                                     ymax1=0.1,
                                     ymax2=100000,
                                     xlabel='Training Step',
                                     ylabel_1='Queue Size (images)',
                                     ylabel_2='Single-Step \n Running Time (sec)',
                                     show_legends=False,
                                     show_xlabel=show_xlabel,
                                     show_ylabel_1=show_ylabel_1,
                                     show_ylabel_2=show_ylabel_2,
                                     annotate_col=annotate_col,
                                     col_annotation=col_annotation,
                                     annotate_row=annotate_row,
                                     row_annotation=row_annotation)

    plt.grid(True,
             zorder=0)

    ax1.legend(bbox_to_anchor=(0.2, 0.05),
               loc="lower left",
               mode="expand",
               bbox_transform=fig.transFigure,
               borderaxespad=0,
               ncol=3)

    ax2.legend(bbox_to_anchor=(0.6, 0.05),
               loc="lower left",
               mode="expand",
               bbox_transform=fig.transFigure,
               borderaxespad=0,
               ncol=3)

    # plt.tight_layout(rect=(0.001, 0.001, 1.0, 1.0))

    fig.set_size_inches(9, 15)

    plt.suptitle(r"Running Time ($\mu\pm\sigma$) and Queue Size vs. Training Step")
    # fig.savefig('C:\\Users\\Justi\\Research\\61\\synaptic_annealing\\figures\\deep_sa_comparative_study.eps',
    #             rasterized=True,
    #             dpi=600,
    #             bbox_inches='tight',
    #             pad_inches=0.05)
    plt.show()




# -------------------------------------------------
# -------------------------------------------------
# -------------------------------------------------
# -------------------------------------------------
# -------------------------------------------------

def batch_interval_vs_threads_plot():

    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True
    matplotlib.rcParams.update({'font.size': 10})

    plt.style.use('seaborn-whitegrid')

    df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/queue_exhaustion_study.csv')
    df = df.sort_values(['batch_interval', 'train_enqueue_threads'])

    df = df.loc[(df.batch_interval != 128)]
    df = df.loc[(df.train_enqueue_threads != 128)]

    row_content = df.train_enqueue_threads
    row_levels = row_content.unique()

    col_content = df.batch_interval
    col_levels = col_content.unique()

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

            run_df = df.loc[(row_content == row_level) &
                            (col_content == col_level)]

            mrt_mean = run_df.groupby(['step_num'])['mean_running_time'].mean().tolist()
            mrt_std = run_df.groupby(['step_num'])['mean_running_time'].std().tolist()

            queue_size_mean = run_df.groupby(['step_num'])['queue_size'].mean().tolist()
            queue_size_std = run_df.groupby(['step_num'])['queue_size'].std().tolist()

            step = run_df['step_num']
            step = run_df.groupby(['step_num'])['step_num'].mean().tolist()

            ax1, ax2 = bar_line_plot(ax1=ax,
                                     time=step,
                                     data1=mrt_mean,
                                     data1bar=mrt_std,
                                     data2=queue_size_mean,
                                     c1='r',
                                     c2='b',
                                     xmin=0,
                                     ymin1=0,
                                     ymin2=0,
                                     xmax=1000,
                                     ymax1=0.1,
                                     ymax2=100000,
                                     xlabel='Training Step',
                                     ylabel_1='Queue Size (images)',
                                     ylabel_2='Single-Step \n Running Time (sec)',
                                     show_xlabel=show_xlabel,
                                     show_ylabel_1=show_ylabel_1,
                                     show_ylabel_2=show_ylabel_2,
                                     annotate_col=annotate_col,
                                     col_annotation=col_annotation,
                                     annotate_row=annotate_row,
                                     row_annotation=row_annotation,
                                     show_legends=False)

            # plt.legend()

    plt.grid(True,
             zorder=0)
    ax1.legend(bbox_to_anchor=(0.37, 0.03),
               loc="lower left",
               mode="expand",
               bbox_transform=fig.transFigure,
               borderaxespad=0,
               ncol=3)

    ax2.legend(bbox_to_anchor=(0.55, 0.025),
               loc="lower left",
               mode="expand",
               bbox_transform=fig.transFigure,
               borderaxespad=0,
               ncol=3)

    # plt.tight_layout(pad=0.0, w_pad=0.0, h_pad=0.0)

    fig.set_size_inches(20, 15)

    plt.suptitle("Queue Size and Single-Step Running Time for Combinations of Batch Interval and Enqueue Threads")
    # fig.savefig('C:\\Users\\Justi\\Research\\61\\synaptic_annealing\\figures\\deep_sa_comparative_study.eps',
    #             rasterized=True,
    #             dpi=600,
    #             bbox_inches='tight',
    #             pad_inches=0.05)
    plt.show()



# -------------------------------------------------
# -------------------------------------------------
# -------------------------------------------------
# -------------------------------------------------

def queue_rate_surface():

    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True
    matplotlib.rcParams.update({'font.size': 12})

    plt.style.use('seaborn-whitegrid')

    df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/queue_exhaustion_study.csv')
    # df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/tensorflow_experiment_merged.csv')

    df = df.sort_values(['batch_interval', 'train_enqueue_threads'])


    df = df.loc[(df.batch_interval != 128)]
    df = df.loc[(df.train_enqueue_threads != 128)]

    df = df.loc[(df.queue_size > 50)]
    df = df.loc[(df.queue_size < 99500)]

    row_content = df.train_enqueue_threads
    row_levels = row_content.unique()

    col_content = df.batch_interval
    col_levels = col_content.unique()

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
                           linewidth=0.0,
                           antialiased=True)

    ax.set_xlabel('Thread Count')

    ax.set_ylabel('Batch Interval')

    ax.set_zlabel('Net Enquque Rate (images/sec)')

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()



# -------------------------------------------------
# -------------------------------------------------
# -------------------------------------------------
# -------------------------------------------------
# -------------------------------------------------

def running_time_surface():

    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True
    matplotlib.rcParams.update({'font.size': 12})

    plt.style.use('seaborn-whitegrid')

    df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/queue_exhaustion_study.csv')
    # df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/tensorflow_experiment_merged.csv')

    df = df.sort_values(['batch_interval', 'train_enqueue_threads'])

    df = df.loc[(df.batch_interval != 128)]
    df = df.loc[(df.train_enqueue_threads != 128)]

    row_content = df.train_enqueue_threads
    row_levels = row_content.unique()

    col_content = df.batch_interval
    col_levels = col_content.unique()

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

    ax.set_xlabel('Thread Count')

    ax.set_ylabel('Batch Interval')

    ax.set_zlabel('Mean Running Time')

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()

# -------------------------------------------------
# -------------------------------------------------
# -------------------------------------------------
# -------------------------------------------------
# -------------------------------------------------
# -------------------------------------------------
# -------------------------------------------------
# -------------------------------------------------


def generalization_stability_plot():

    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rcParams['text.latex.unicode'] = True
    matplotlib.rcParams.update({'font.size': 12})

    plt.style.use('seaborn-whitegrid')

    df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/queue_exhaustion_study.csv')
    # df = pd.read_csv('C:/Users/Justi/Research/log/queue_exhaustion/tensorflow_experiment_merged.csv')

    df = df.sort_values(['batch_interval', 'train_enqueue_threads'])

    df = df.loc[(df.step_num > 30)]
    df = df.loc[(df.pause_time == 10)]

    df = df.loc[(df.batch_interval != 128)]
    df = df.loc[(df.train_enqueue_threads != 128)]
    # df = df.loc[(df.train_enqueue_threads == 64)]

    row_content = df.batch_interval
    row_levels = row_content.unique()

    mean_running_times_means = []
    mean_running_times_stds = []
    val_loss_means = []
    val_loss_stds = []

    for i, row_level in enumerate(row_levels):

        run_df = df.loc[(row_content == row_level)]

        val_loss_mean = run_df.groupby(['step_num'])['val_loss'].mean().tolist()
        val_loss_std = run_df.groupby(['step_num'])['val_loss'].std().tolist()

        val_loss_means.append(val_loss_mean[-1])
        val_loss_stds.append(val_loss_std[-1])

        mean_running_time_mean = run_df['mean_running_time'].mean()
        mean_running_time_std = run_df['mean_running_time'].std()

        mean_running_times_means.append(mean_running_time_mean)
        mean_running_times_stds.append(mean_running_time_std)

    fig = plt.figure()
    ax = fig.gca()


    # ax = fig.gca(projection='3d')

    # surf = ax.plot_trisurf(X,
    #                        Y,
    #                        mean_running_times,
    #                        cmap=cm.coolwarm,
    #                        linewidth=0.1,
    #                        antialiased=True)


    # fig.colorbar(surf, shrink=0.5, aspect=5)


    for i, row_level in enumerate(row_levels):

        x = val_loss_means[i]
        y = mean_running_times_means[i]

        xerr = val_loss_stds[i]
        yerr = mean_running_times_stds[i]

        plt.scatter(x, y, color='black')

        plt.errorbar(x, y, xerr, yerr, color='black')

        ax.annotate(r'Batch Interval $ = ' + str(row_level) + '$', (x, y))


    ax.set_xlabel('Validation Loss')
    ax.set_ylabel('Running Time')
    # plt.legend()
    plt.show()


# Validation set error vs. speedup. Each point a different marker (I_B).

# -------------------------------------------------
# -------------------------------------------------
# -------------------------------------------------

single_plot_example()
thread_count_col()
batch_interval_vs_threads_plot()
queue_rate_surface()
running_time_surface()
generalization_stability_plot()
