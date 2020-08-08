from solver import *
from geneticalgorithm import geneticalgorithm as ga
import numpy as np


class GeneticSolver(Solver):
    def __init__(self, pictures, fitness, population_size, num_generations):
        self.algorithm_param = {
                                'max_num_iteration': num_generations,
                                'population_size': population_size,
                                'mutation_probability': 0.1,
                                'elit_ratio': 0.01,
                                'crossover_probability': 0.5,
                                'parents_portion': 0.3,
                                'crossover_type': 'uniform',
                                'max_iteration_without_improv': None
                                }
        self.model = ga(function=fitness,
                        dimension=1000000,
                        variable_type='bool',
                        )

    def generate_initial_population(self):
        pass

    def solve(self):
        self.model.run()

    def get_answer(self):
        return self.model.output_dict[0]

    def get_answer_loss(self):
        return self.model.output_dict[1]
        # return self.model.report what does it do check.


def f(X):
    return np.sum(X)


import time
#
# start = time.time()
# model = ga(function=f, dimension=100, variable_type='bool')
# model.run()
# print("runtime was : ", time.time() - start)

