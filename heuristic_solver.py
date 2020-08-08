from solver import *


class HeuristicSolver(Solver):
    def __init__(self, pictures, fitness_func, successor_func, generate_initial_func,
                 num_iteration):
        self._pictures = pictures
        self._get_fitness = fitness_func
        self._get_successor = successor_func
        self._initial_state_generator = generate_initial_func
        self._current_state = generate_initial_func()
        self._current_score = fitness_func(self._current_state.get_matrix())
        self._num_iterations = num_iteration

    def solve(self):
        pass

    def get_answer(self):
        pass

    def get_answer_loss(self):
        pass