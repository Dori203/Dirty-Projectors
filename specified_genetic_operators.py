import numpy as np
from image_processing import merge_matrices
import random
from image_processing import prob_binary_op, flip_random_pixels, change_neighbourhood
from GUI import print_3d_matrix


# ========================== wrapper for 1D operators ===========================

def cx_1D_operator(icls):
    """creates a 1D decorator for a deap package crossover operation
    @param icls - individual class (used as a constructor)
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            shape = args[0].shape
            ind_1 = args[0].reshape(shape[0] ** 3)
            ind_2 = args[1].reshape(shape[0] ** 3)
            res1, res2 = func(ind_1, ind_2, **kwargs)
            return icls(res1.reshape(shape)), icls(res2.reshape(shape))

        return wrapper

    return decorator


def mut_1D_operator(icls):
    """creates a 1D decorator for a deap package mutator operation
    @param icls - individual class (used as a constructor)
    """

    def decorator(func):
        def wrapper(*args, **kwargs):
            shape = args[0].shape
            ind = args[0].reshape(shape[0] ** 3)
            res = np.array(func(ind, **kwargs))
            return icls(res.reshape(shape)),

        return wrapper

    return decorator


# INIT
# ========================== individual 3d initializers ============================


def init_3d_individual(icls, shape, black_ratio=0.01):
    """
    this init function based on the problem domain - it's preferred to use relatively
    small amount of black pixel to start with, beacuse reducing their amount is
    difficult and requires a lot of generations.
    @param icls: individual class defined for the problem.
    @param shape:
    @return:
    """

    ind = np.where(np.random.rand(*shape) < black_ratio, 1, 0)
    # print_3d_matrix(ind, f"blacks={np.sum(ind)}", 32)
    return icls(ind)


def init_population_exploit(icls, solutions=()):
    for sol in solutions:
        pass
    # out of scope for right now - need to check various functions for mating the given solution
    # to create sufficient size population.


# CROSSOVER
# =========================== 3d crossover ==============================
# todo: does every crossover called and i need to implement the cx prob mechanism?


def cx_3d_single_cut(ind_1, ind_2, icls):
    """
    the x, y, z of the cutting point are generated. The cutting point divides the
    chromosomes into 8 parts denoted {1,..,8}. The first offspring inherits the genes
    from parts 1, 3, 5, 7 of the first parent, and parts 2, 4, 6, 8 from the second
    parent. The second offspring inherits the genes in reverse manner.
    @param ind_1:
    @param ind_2:
    @param icls:
    @return:
    """
    offspring_1 = ind_1[:]
    offspring_2 = ind_2[:]
    blocks = get_blocks_single_cut_3d(ind_1.shape[0])
    for idx, block in enumerate(blocks):
        if idx % 2 == 0:
            offspring_1 = merge_matrices(offspring_1, ind_2, *block)
            offspring_2 = merge_matrices(offspring_2, ind_1, *block)
    return icls(offspring_1), icls(offspring_2)


def cx_block_uniform(ind_1, ind_2, icls, probability=0.1):
    """
    divides the the parental chromosomes into i x j x k blocks of genes with randomly
    generated dimensions. offspring is generated from exchange of blocks from the first
    parent with the corresponding blocks of the other parent, according to a given
    probability.

    @param ind_2:
    @param ind_1:
    @param icls:
    @param probability:
    @return:
    """
    offspring_1 = ind_1[:]
    offspring_2 = ind_2[:]
    blocks = get_blocks_single_cut_3d(ind_1.shape[0])
    for block in blocks:
        if random.random() < probability:
            offspring_1 = merge_matrices(offspring_1, ind_2, *block)
            offspring_2 = merge_matrices(offspring_2, ind_1, *block)
    # print_3d_matrix(ind, f"blacks={np.sum(ind)}", 32)
    return icls(offspring_1), icls(offspring_2)


def cx_rectangle_style(ind1, ind2, icls):
    """
    a small rectangle from the first parent is selected. genes from this rectangle are
    copied to the offspring and remaining genes are inherited from the other parent.
    @param ind1:
    @param ind2:
    @param icls:
    @return:
    """
    indices = get_random_sub_cube(ind1.shape[0])
    offspring_1 = merge_matrices(ind1, ind2, *indices)
    offspring_2 = merge_matrices(ind2, ind1, *indices)
    return icls(offspring_1), icls(offspring_2)


def cx_layer_swap(ind_1, ind_2, icls):
    """
    Two parents swap a whole axis between each other.
    @param ind_1:
    @param ind_2:
    @param icls:
    @return:
    """
    axis = np.random.randint(3)
    if axis == 0:
        x, y, z = get_cut_points(ind_1.shape[0], 0, 1, 1)
        offspring_1 = merge_matrices(ind_1, ind_2, 0, ind_1.shape[0], y, y, z, z)
        offspring_2 = merge_matrices(ind_2, ind_1, 0, ind_1.shape[0], y, y, z, z)
    elif axis == 1:
        x, y, z = get_cut_points(ind_1.shape[0], 1, 0, 1)
        offspring_1 = merge_matrices(ind_1, ind_2, x, x, 0, ind_1.shape[0], z, z)
        offspring_2 = merge_matrices(ind_2, ind_1, x, x, 0, ind_1.shape[0], z, z)
    else:
        x, y, z = get_cut_points(ind_1.shape[0], 1, 1, 0)
        offspring_1 = merge_matrices(ind_1, ind_2, x, x, y, y, 0, ind_1.shape[0])
        offspring_2 = merge_matrices(ind_2, ind_1, x, x, y, y, 0, ind_1.shape[0])
    return icls(offspring_1), icls(offspring_2)


def cx_corner_block(ind1, ind2, icls):
    """
    In 3D chromosomal space the point with coordinates i, j and k is generated.
    The offspring inherits the genes that are at the coordinates x < i, y < j and z < k from
    one parent, the remaining genes are from the other parent.

    @param ind1:
    @param ind2:
    @param icls:
    @return:
    """
    upper_bound = ind1.shape[0]
    x_cut_point, y_cut_point, z_cut_point = get_cut_points(upper_bound, 1, 1, 1)

    offspring_1 = merge_matrices(ind1, ind2, 0, x_cut_point, 0, y_cut_point, 0, z_cut_point)
    offspring_2 = merge_matrices(ind2, ind1, 0, x_cut_point, 0, y_cut_point, 0, z_cut_point)
    return icls(offspring_1), icls(offspring_2)


def cx_geographical():
    # todo - hadas
    pass


def cx_random_operator():
    """
    . Mixed crossover: T he ope rator, taken from the above five types, is
    determined randomly.
    @return:
    """
    operator = []


# ========================== 2D crossover ==========================
# not in our scope for now


# MUTATION
# ======================== 3D Mutation operators ====================

def mut_invert_sub_cube(ind, icls):
    """
    inverts the values of the genes in a random generated sub-cube.
    @param ind:
    @param icls
    @return:
    """
    flipped = 1 - ind[:]
    indices = get_random_sub_cube(ind.shape[0])
    return icls(merge_matrices(ind, flipped, *indices)),


def mut_shuffle_along_axis(ind, icls):
    """
    shuffles values of genes along a randomly chosen axis
    @param ind:
    @param indpb:
    @param icls:
    @return:
    """
    axis = random.randint(0, 2)
    idx = np.random.rand(*ind.shape).argsort(axis=axis)
    return icls(np.take_along_axis(ind, idx, axis=axis)),


def mut_prob_binary_op(ind, icls, length=2):
    """
    see doc of the called function
    @param ind:
    @param icls:
    @param length:
    @return:
    """
    return icls(prob_binary_op(ind, length)),


def mut_flip_random_pixel(ind, icls, num_pixels=3):
    """
    see doc of the called function
    @param ind:
    @param icls:
    @param num_pixels:
    @return:
    """
    return icls(flip_random_pixels(ind, num_pixels)),


def mut_change_neighbourhood(ind, icls, flip_prob=0.5):
    """
    see doc of the called function
    @param ind:
    @param icls:
    @param flip_prob:
    @return:
    """
    return icls(change_neighbourhood(ind, flip_prob)),


methods = {0: mut_change_neighbourhood,
           1: mut_flip_random_pixel,
           2: mut_prob_binary_op}


def mut_enhanced(ind, icls, prob=0.7):
    method_probabilities = [prob, 6 * (1 - prob) / 7, (1 - prob) / 7]
    method_key = np.random.choice(3, p=method_probabilities)
    res, = methods[method_key](ind, icls)
    if len(res.shape) != 3:
        print("hi")
    return icls(res),


# ============================ helpers ===============================

def get_blocks_single_cut_3d(upper_bound):
    """
    divides a 3d cube to 8 blocks
    @param upper_bound: range upper bound for the coordinates.
    @param cut_points: (x,y,z)
    @return: list of blocks indices.
    """
    x_cut_point, y_cut_point, z_cut_point = get_cut_points(upper_bound, 1, 1, 1)
    blocks = [
        # lower z cut
        (0, x_cut_point, 0, y_cut_point, 0, z_cut_point),
        (x_cut_point, upper_bound, 0, y_cut_point, 0, z_cut_point),
        (x_cut_point, upper_bound, y_cut_point, upper_bound, 0, z_cut_point),
        (x_cut_point, upper_bound, y_cut_point, upper_bound, 0, z_cut_point),
        # upper z cut
        (0, x_cut_point, 0, y_cut_point, z_cut_point, upper_bound),
        (x_cut_point, upper_bound, 0, y_cut_point, z_cut_point, upper_bound),
        (x_cut_point, upper_bound, y_cut_point, upper_bound, z_cut_point, upper_bound),
        (x_cut_point, upper_bound, y_cut_point, upper_bound, z_cut_point, upper_bound)
    ]
    return blocks


def get_cut_points(upper_bound, x, y, z):
    x_cut_point = random.randrange(1, upper_bound - 1) if x else 0
    y_cut_point = random.randrange(1, upper_bound - 1) if y else 0
    z_cut_point = random.randrange(1, upper_bound - 1) if z else 0
    return x_cut_point, y_cut_point, z_cut_point


def get_random_sub_cube(ind_size):
    max_dim = round(ind_size / 10)
    start_x, start_y, start_z = random.randrange(1, ind_size), \
                                random.randrange(1, ind_size), \
                                random.randrange(1, ind_size)

    end_x, end_y, end_z = random.randrange(start_x, start_x + max_dim), \
                          random.randrange(start_y, start_y + max_dim), \
                          random.randrange(start_z, start_z + max_dim)
    return start_x, end_x, start_y, end_y, start_z, end_z
