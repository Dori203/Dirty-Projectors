from solver import *
import numpy as np
import constraint


def extract_pics(mat):
    return (np.sum(mat, axis=0) > 0), (np.sum(mat, axis=1) > 0), (np.sum(mat, axis=2) > 0)


has = lambda *row: sum(row) > 0
hasnt = lambda *row: sum(row) == 0
_axis = [lambda row, col, dim: (row + col * dim + i * pow(dim, 2) for i in range(dim)),
         lambda row, col, dim: (row + col * pow(dim, 2) + i * dim for i in range(dim)),
         lambda row, col, dim: (row * dim + col * pow(dim, 2) + i for i in range(dim))]


class CSPSolver(Solver):
    def __init__(self, pics, solver=None):
        if solver == 'RecursiveBacktrackingSolver':
            solver = constraint.RecursiveBacktrackingSolver()
        elif solver == 'MinConflictsSolver':
            solver = constraint.MinConflictsSolver()
        else:
            solver = constraint.BacktrackingSolver()
        self.dim = pics[0].shape[0]
        self.problem = constraint.Problem(solver)
        self.problem.addVariables(range(pow(self.dim, 3)), [0, 1])
        for i, pic in enumerate(pics):
            self._add_constraints(pic, _axis[i])
        solution = self.problem.getSolution()
        if solution is None:
            self.ans = None
        else:
            self.ans = np.array([solution[i] for i in range(pow(self.dim, 3))]).reshape((self.dim, self.dim, self.dim))

    def _add_constraints(self, pic, calc_row):
        for row in range(self.dim):
            for col in range(self.dim):
                if pic[col, row]:
                    self.problem.addConstraint(has, tuple(calc_row(row, col, self.dim)))
                else:
                    self.problem.addConstraint(hasnt, tuple(calc_row(row, col, self.dim)))

    def get_answer(self):
        return self.ans


class MRV(Solver):

    def __init__(self, pics):
        self.pics = pics
        self.dim = pics[0].shape[0]
        self.answer = np.ones(pow(self.dim, 3))
        for i, pic in enumerate(pics):
            self._add_constraints(pic, _axis[i])
        self.answer = self.answer.reshape((self.dim, self.dim, self.dim))
        sides = extract_pics(self.answer)
        for i, pic in enumerate(pics):
            if not (pic == sides[i]).all():
                self.answer = None

    def _add_constraints(self, pic, calc_row):
        for row in range(self.dim):
            for col in range(self.dim):
                if not pic[col, row]:
                    for i in calc_row(row, col, self.dim):
                        self.answer[i] = 0

    def get_answer(self):
        return self.answer
