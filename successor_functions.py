import random
from state import State
import numpy as np
import image_processing as impr

UPDATE_NEIGHBOURHOOD = 0
RANDOMLY_FLIP_PIXELS = 1
BINARY_OPERATION = 2


# choosing the best dot to change out of 30 random dots
def naive_successors(curr_state, num_of_neighbors=30):
    """naive approach, random choice of fixed number of pixels to flip."""
    neighbors = [(random.randrange(0, curr_state.get_dim()),
                  random.randrange(0, curr_state.get_dim()),
                  random.randrange(0, curr_state.get_dim())) for i in
                 range(num_of_neighbors)]
    states = [curr_state.get_matrix().copy() for i in range(num_of_neighbors + 1)]
    # switch the dot
    for i in range(num_of_neighbors):
        states[i][neighbors[i]] = 1 - states[i][neighbors[i]]

    return [State(states[i]) for i in range(num_of_neighbors + 1)]


def neighbouring_successors(curr_state, probability_of_neighbour=0.85, num_of_neighbors=30, switch_pixels=3):
    result_matrices = neighbouring_successors_matrix(curr_state.get_matrix(), probability_of_neighbour,
                                                     num_of_neighbors, switch_pixels)
    return [State(neighbour_matrix) for neighbour_matrix in result_matrices]


def line_successor(curr_state, neighbour_degree=4, change_line_prob=0.05, random_prob=0.05):
    result_matrices = []
    method = np.random.choice(3, 1, p=[1 - random_prob, 3 * random_prob / 4, 1 * random_prob / 4])
    if impr.count_black_pixels(curr_state.get_matrix()) == 0 or method == 1:
        return naive_successors(curr_state)
    else:
        result_matrices = impr.line_successor(curr_state.get_matrix(), neighbour_degree, change_line_prob)
    if method == 2:
        result_matrices.append(impr.prob_binary_op(curr_state.get_matrix()))
    return [State(neighbour_matrix) for neighbour_matrix in result_matrices]


def neighbouring_successors_matrix(three_d_matrix, probability_of_neighbour, num_of_neighbors, switch_pixels=3):
    """
    Each modification can by a random pixel swap, or a change of pixel 2*2*2 block, around an already black pixel.
    @param curr_state: current state.
    @param probability_of_neighbour: probability of modifing a block around an existing pixel.
    @param num_of_neighbors: Amount of successors in every iteration.
    @return: All newly created successors.
    """
    # create neighbours
    neighbour_matrices = [three_d_matrix.copy() for i in range(num_of_neighbors)]
    update_method_array = np.random.choice(3, num_of_neighbors,
                                           p=[probability_of_neighbour, 4 * (1 - probability_of_neighbour) / 5,
                                              1 * (1 - probability_of_neighbour) / 5])

    for i, update_method in enumerate(update_method_array):
        if update_method == UPDATE_NEIGHBOURHOOD:
            # change a specific pixels neighbourhood.
            neighbour_matrices[i] = impr.change_neighbourhood(neighbour_matrices[i])
        elif update_method == 1:
            # Randomly switch the dot for switch_pixels pixels.
            neighbour_matrices[i] = impr.flip_random_pixels(neighbour_matrices[i])
        else:
            # perform a binary operator.
            neighbour_matrices[i] = impr.prob_binary_op(neighbour_matrices[i])
    return neighbour_matrices


def change_layer(x):
    """
    given a chunk of a matrix, chooses at random pixels and flips them.
    :param x: given chunk of the matrix
    """
    a = np.random.binomial(1, 0.001, x.shape)
    choice = np.where(a == 1)  # todo check what is the best probability
    x[choice] = 1 - x[choice]


def layer(mat, layer_num):
    """
    given a state and number of 'onion' layer to flip. chooses at random pixels from the layer and flips them.
    *note* the dimension must be even
    """
    dim = mat.shape[0]
    ind = int(dim / 2)
    change_layer(mat[ind - 1 - layer_num:ind + 1 + layer_num:1 + 2 * layer_num, ind - 1 - layer_num:ind + 1 + layer_num,
                 ind - 1 - layer_num:ind + 1 + layer_num])
    change_layer(mat[ind - layer_num:ind + layer_num, ind - 1 - layer_num:ind + 1 + layer_num:1 + 2 * layer_num,
                 ind - 1 - layer_num:ind + 1 + layer_num])
    change_layer(mat[ind - layer_num:ind + layer_num, ind - layer_num:ind + layer_num,
                 ind - 1 - layer_num:ind + 1 + layer_num:1 + 2 * layer_num])


def peel_an_onion(curr_state, in_out=0):
    """
    given a state, chooses at random an 'onion' layer and then flips random pixels from the chosen layer
    """
    dim = curr_state.shape[0]
    ind = int(dim / 2)
    layer_num = min(int(abs(np.random.normal(0, ind - 1))), ind - 1)
    # layer_num = min(ind, np.random.poisson())
    layer(curr_state, layer_num)
    return curr_state


def onion(curr_state, num_of_neighbors=30):
    """
    given a state, creates neighbors where each neighbor has one 'onion' layer that has some random flipped pixels
    :param num_of_neighbors: number of neighbors to return
    :return: a list of neighbors
    """
    return [State(peel_an_onion(curr_state.get_matrix().copy())) for i in range(num_of_neighbors)]


def flip(mat, p=0.0001):
    """give each dot a positive probability to change"""
    # todo: check both ways: uniform distribution and bernouli distribution with random parameter.
    # two approaches for using this methos - compute random cube, or sub-cubes for recursive solution.
    choice = np.where(np.random.binomial(1, p, mat.shape) == 1)
    mat[choice] = 1 - mat[choice]
    return mat


def flip_all(curr_state, num_of_neighbors=30):
    return [State(flip(curr_state.get_matrix().copy())) for i in range(num_of_neighbors)]


def column_row_child(curr_state, num_of_neighbors=30):
    """ at each iteration this function will provide neighbors by blanking a line parallel to the one of the 3 axis."""
    matrix = curr_state.get_matrix()

    neighbors = []

    for i in range(int(num_of_neighbors / 3)):
        row = (random.randrange(0, curr_state.get_dim()), random.randrange(0, curr_state.get_dim()))
        col = (random.randrange(0, curr_state.get_dim()), random.randrange(0, curr_state.get_dim()))
        line = (random.randrange(0, curr_state.get_dim()), random.randrange(0, curr_state.get_dim()))
        new_matrix_x = np.zeros(matrix.shape)
        new_matrix_x[:, :, :] = matrix[:, :, :]
        new_matrix_x[row[0]:row[0] + 1, row[1]:row[1] + 1, :] = 0
        new_matrix_y = np.zeros(matrix.shape)
        new_matrix_y[:, :, :] = matrix[:, :, :]
        new_matrix_y[:, col[0]:col[0] + 1, col[1]:col[1] + 1] = 0
        new_matrix_z = np.zeros(matrix.shape)
        new_matrix_z[:, :, :] = matrix[:, :, :]
        new_matrix_z[line[0]:line[0 + 1], :, line[1]:line[1] + 1] = 0
        neighbors.append(State(new_matrix_x))
        neighbors.append(State(new_matrix_y))
        neighbors.append(State(new_matrix_z))
    return neighbors
