import matplotlib.pyplot as plt
import numpy as np

from matplotlib.ticker import MultipleLocator, FormatStrFormatter

from typing import List


def plot():
    std_pi_path = 'standardRL_policy.csv'
    iRL_pi_path = 'iRL_policy.csv'

    paths = std_pi_path, iRL_pi_path

    titles = ['Standard PPO policy', 'iRL PPO policy']

    legends = [draw_plot_plt(path, titles[i], False)
               for i, path in enumerate(paths)]

    plt.suptitle('Comparison of standard RL Policy and Inverse RL Policy')
    plt.title('IRL uses expert data collected over 250 policy updates',  size=8)
    plt.legend(handles=legends)
    plt.xlabel('step (s)')
    plt.ylabel('Average return')

    plt.show()


def draw_plot(axis, title, path=None, figures=None, change_ticks=False,
              sci_ticks=True, smooth=False):
    """
        Draws plots using loaded csv data
    """
    title = title.replace('_', ' ')
    figures = np.loadtxt(path, delimiter=',',
                         skiprows=1) if not np.any(figures) else figures
    x = figures[:, 1]
    y = figures[:, 2]
    x_ = x
    y_ = y

    if smooth:
        from scipy.interpolate import make_interp_spline
        x_ = np.linspace(x.min(), x.max(), 300000)
        Bs_pline = make_interp_spline(x, y)
        y_ = Bs_pline(x_)

    axis.plot(x_, y_, label=title)
    axis.set_title(title, fontdict={'size': 10})
    axis.set_ylabel('reward')
    axis.set_xlabel('step (s)')

    if sci_ticks:
        axis.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    if change_ticks:
        axis.yaxis.set_major_locator(MultipleLocator(100))
        axis.yaxis.set_major_formatter(FormatStrFormatter('%d'))
        axis.yaxis.set_minor_locator(MultipleLocator(20))

    axis.grid()


def draw_plot_plt(path, title, sci_ticks=True, smooth=True):
    """
        Draw non-axis plots
    """
    figures = np.loadtxt(path,
                         delimiter=',',
                         skiprows=1,
                         )
    x = figures[:, 1]
    y = figures[:, 2]
    x_ = x
    y_ = y

    if smooth:
        x_ = exp_smooth(x)
        y_ = exp_smooth(y)

    legend = plt.plot(x_, y_, label=title)
    if sci_ticks:
        plt.ticklabel_format(axis='x', style='sci', scilimits=(0, 0))

    return legend[0]


# Weight between 0 and 1
def exp_smooth(scalars: List[float], weight: float = 0.8) -> List[float]:
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + \
            (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)                        # Save it
        # Anchor the last smoothed value
        last = smoothed_val

    return smoothed


if __name__ == '__main__':
    plot()
