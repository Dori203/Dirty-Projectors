import random
from state import State

# choosing the best dot to change out of 30 random dots
"""temporal approach - for testing the num_of_neighbors parameter"""


def naive_successors(curr_state, num_of_neighbors=30):
    """naive approach, random choice of fixed number of pixels to flip.
    todo: might be needed to change for support in rgb photos."""

    neighbors = [(random.randrange(0, curr_state.get_dim()),
                  random.randrange(0, curr_state.get_dim()),
                  random.randrange(0, curr_state.get_dim())) for i in
                 range(num_of_neighbors)]
    states = [curr_state.get_matrix().copy() for i in range(num_of_neighbors+1)]
    # switch the dot
    for i in range(num_of_neighbors):
        states[i][neighbors[i]] = 1 - states[i][neighbors[i]]

    # scores = [self._fitness(State(states[i]), self._pictures) for i in range(NUM_OF_NEIGHBORS)]
    # min_idx = scores.index(min(scores))
    return [State(states[i]) for i in range(num_of_neighbors+1)]


def peel_an_onion(self, curr_state, layer_num, in_out=0):
    """given a 3d matrix, every successor is generated as a step towards the inner cube.
    in other words, we choose randomly the values of pixels only in one 3d layer of the cube.
    in more other words - peel layers until you get the result. the randomness is to diminish the number of
    successors which is still big. """
    # todo: allow progess from inner cube outside and the opposite.
    # this process should support each initial state - blank cube (add only black pixels), black cube (only
    # delete pixels), and random valued cube.
    pass


def flip_all(self, curr_state):
    """give each dot a positive probability to change"""
    # todo: check both ways: uniform distribution and bernouli distribution with random parameter.
    # two approaches for using this methos - compute random cube, or sub-cubes for recursive solution.
