import matplotlib.pyplot as plt


def plot(x, y, fmt='o-', xlabel=None, ylabel=None, title=None, grid=False):
    """

    :param x:
    :param y:
    :param fmt:
    :param xlabel:
    :param ylabel:
    :param title:
    :param grid:
    :return:
    """
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, fmt)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(grid)
    plt.show()

    return plt
