import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D


def print_3d_matrix(three_d_matrix):
    x, y, z = np.where(three_d_matrix != 0)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(x, y, z, c='blue')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.xlim(0, 40)
    plt.ylim(0, 40)
    plt.title("pyramid -120, 0, 120 rotations 1000 iter")
    plt.title("pyramid only front and bottom")
    fig.show()
