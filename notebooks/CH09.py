import numpy as np
import matplotlib.pylab as plt
import tensorflow as tf


def example_plot(xy, labels, a, b, title, filename=None):
    # Shape
    c_shape = ['bs', 'r^']

    # 1. Point
    for k, (point, label) in enumerate(zip(xy, labels),1):
        x,y = point
        plt.plot(x, y, c_shape[label[0]], mec='k')
        plt.text(x, y, '$P_{}$'.format(k), size=15, \
                 verticalalignment='top', horizontalalignment='left')

    # 2. Decision line
    tmp = np.linspace(0,1,500)
    decision_line = a * tmp + b
    plt.plot(tmp, decision_line, 'k-', linewidth=3)

    # 3. Contour
    X, Y = np.meshgrid(tmp, tmp)
    Z = np.zeros_like(X)
    Z[Y > a * X + b] = 1

    plt.contourf(X, Y, Z, cmap='coolwarm')
    # plt.axis('equal')
    plt.grid(linestyle='--', alpha=0.5)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title(title)
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()


def example_plot_only_line(xy, labels, a, b, title, filename=None):
    # Shape
    c_shape = ['bs', 'r^']

    # 1. Point
    for k, (point, label) in enumerate(zip(xy, labels),1):
        x,y = point
        plt.plot(x, y, c_shape[label[0]], mec='k')
        plt.text(x, y, '$P_{}$'.format(k), size=15, \
                 verticalalignment='top', horizontalalignment='left')

    # 2. Decision line
    tmp = np.linspace(0,1,500)
    decision_line = a * tmp + b
    plt.plot(tmp, decision_line, 'k-', linewidth=3)

    # 3. Contour
#     X, Y = np.meshgrid(tmp, tmp)
#     Z = np.zeros_like(X)
#     Z[Y > a * X + b] = 1

#     plt.contourf(X, Y, Z, cmap='coolwarm')
    # plt.axis('equal')
    plt.grid(linestyle='--', alpha=0.5)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title(title)
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

    
def example_plot_wo_contour(xy, labels, title, filename=None):
    # Shape
    c_shape = ['bs', 'r^']

    # 1. Point
    for k, (point, label) in enumerate(zip(xy, labels),1):
        x,y = point
        plt.plot(x, y, c_shape[label[0]], mec='k')
        plt.text(x, y, '$P_{}$'.format(k), size=15, \
                 verticalalignment='top', horizontalalignment='left')

#     # 2. Decision line
#     tmp = np.linspace(0,1,500)
#     decision_line = a * tmp + b
#     plt.plot(tmp, decision_line, 'k-', linewidth=3)

    # 3. Contour
#     X, Y = np.meshgrid(tmp, tmp)
#     Z = np.zeros_like(X)
#     Z[Y > a * X + b] = 1

#     plt.contourf(X, Y, Z, cmap='coolwarm')
    # plt.axis('equal')
    plt.grid(linestyle='--', alpha=0.5)
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.title(title)
    if filename:
        plt.savefig(filename, dpi=300)
    plt.show()

