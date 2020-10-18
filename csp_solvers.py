from solver import *
from image_processing import *
import numpy as np
import constraint
from state import State

_axis = [lambda row, col, dim: (row + col * dim + i * pow(dim, 2) for i in range(dim)),
         lambda row, col, dim: (row + col * pow(dim, 2) + i * dim for i in range(dim)),
         lambda row, col, dim: (row * dim + col * pow(dim, 2) + i for i in range(dim))]


class CSPSolver(Solver):
    def __init__(self, image_processor: ImageProcessor):
        self.processor = image_processor
        self.dim = image_processor.dim[0]
        self.problem = constraint.Problem(constraint.MinConflictsSolver(steps=self.dim))
        self.problem.addVariables(range(pow(self.dim, 3)), [0, 1])
        for i, pic in enumerate(image_processor.get_images()):
            self._add_constraints(pic, _axis[i])
        self.ans = np.ones(pow(self.dim, 3))

    def solve(self):
        solution = self.problem.getSolution()
        self.ans = np.array([solution[i] for i in range(pow(self.dim, 3))]).reshape((self.dim, self.dim, self.dim))

    def get_num_iters(self):
        return self.dim

    def _add_constraints(self, pic, calc_row):
        for row in range(self.dim):
            for col in range(self.dim):
                if pic[col, row]:
                    self.problem.addConstraint(constraint.MinSumConstraint(1), tuple(calc_row(row, col, self.dim)))
                else:
                    self.problem.addConstraint(constraint.MaxSumConstraint(0), tuple(calc_row(row, col, self.dim)))

    def get_answer(self):
        return State(self.ans)

    def get_answer_loss(self):
        return self.processor.loss_2(self.ans)


class MRV(Solver):

    def __init__(self, image_processor: ImageProcessor):
        self.processor = image_processor
        self.pics = image_processor.get_images()
        self.dim = image_processor.dim[0]
        self.answer = np.ones(pow(self.dim, 3))

    def solve(self):
        for i, pic in enumerate(self.pics):
            self._add_constraints(pic, _axis[i])
        self.answer = self.answer.reshape((self.dim, self.dim, self.dim))

    def get_num_iters(self):
        return 1

    def _add_constraints(self, pic, calc_row):
        for row in range(self.dim):
            for col in range(self.dim):
                if not pic[col, row]:
                    for i in calc_row(row, col, self.dim):
                        self.answer[i] = 0

    def get_answer(self):
        return State(self.answer)

    def get_answer_loss(self):
        return self.processor.loss_2(self.answer)
