from Solver import *
from GUI import *
from constraint import * # https://github.com/python-constraint/python-constraint

pic1 = np.array([[False, True, False], [True, False, True], [False, True, False]])
pic2 = np.array([[True, True, True], [True, False, True], [False, True, False]])
pic3 = np.array([[True, True, True], [False, True, False], [True, False, True]])
sol = np.array(
    [[[0, 1, 0], [1, 0, 1], [0, 1, 0]], [[0, 0, 0], [1, 0, 1], [0, 0, 0]], [[0, 1, 0], [0, 0, 0], [0, 1, 0]]])



smile = np.array([[0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                  [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                  [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                  [1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1],
                  [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                  [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                  [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                  [1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                  [1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                  [1, 1, 0, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1],
                  [0, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0],
                  [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0],
                  [0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]])
heart = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]])
mail = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
                 [1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1],
                 [1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1],
                 [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
                 [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
                 [1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],
                 [1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1],
                 [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                 [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])
cross = np.array([[0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0]])
mt = np.ones(256).reshape(16, 16)

# def extract_pics(mat):
#     return (np.sum(mat, axis=0) > 0), (np.sum(mat, axis=1) > 0), (np.sum(mat, axis=2) > 0)
#
#
# class Variable:
#     def __init__(self, x, y, z, val):
#         self.x = x
#         self.y = y
#         self.z = z
#         self.val = val
#
#
# class Domain:
#     def __init__(self, val):
#         self.val = val
#
#
# class Constraint:
#     def __init__(self, domains, constraint):
#         self.domains = domains
#         self.constraint = constraint
#
#
# class CSPSolver(Solver):
#     def __init__(self, pictures):
#         super().__init__(pictures)
#         self.problem = Problem()
#         self.dim = pictures[0].shape[0]
#         self.problem.addVariables(
#             [(int(i / pow(self.dim, 2)), int(i / self.dim) % self.dim, i % self.dim) for i in range(pow(self.dim, 3))],
#             [True, False])
#         self.pic0 = pictures[0]
#         self.pic1 = pictures[1]
#         self.pic2 = pictures[2]
#         self.mat = np.zeros((self.dim, self.dim, self.dim))
#         # print(1 if self._is_solution(sol) else 0)
#
#     def _solve_helper(self, mat, i):
#         if self._unsolveable(mat):
#             print(mat, "::-")
#             return False
#         if i >= pow(self.dim, 3):
#             # print(mat, "::+")
#             if self._is_solution(mat):
#                 self.mat = mat
#                 return True
#             return False
#         if self._solve_helper(mat, i + 1):
#             return True
#         mat[int(i / pow(self.dim, 2)), int(i / self.dim) % self.dim, i % self.dim] = True
#         return self._solve_helper(mat, i + 1)
#
#     def solve(self):
#         if self._solve_helper(np.zeros((self.dim, self.dim, self.dim)), 0):
#             return self.mat
#         return np.zeros((self.dim, self.dim, self.dim))
#
#     def _is_solution(self, mat):
#         side0, side1, side2 = extract_pics(mat)
#         return (side0 == self.pic0).all() and (side1 == self.pic1).all() and (side2 == self.pic2).all()
#
#     def _unsolveable(self, mat):
#         side0, side1, side2 = extract_pics(mat)
#         return ((self.pic0.astype(int) - side0.astype(int)) < 0).any() or (
#                 (self.pic1.astype(int) - side1.astype(int)) < 0).any() or (
#                        (self.pic2.astype(int) - side2.astype(int)) < 0).any()
has = lambda *row: sum(row) > 0
hasnt = lambda *row: sum(row) == 0


class CSPSolver:
    def __init__(self, pics):
        self.pics = pics
        self.dim = pics[0].shape[0]
        self.problem = Problem()
        self.variables = range(pow(self.dim, 3))
        self.domain = [0, 1]
        self.problem.addVariables(self.variables, self.domain)
        self.axis1 = lambda row, col: tuple(row * pow(self.dim, 2) + col * self.dim + i for i in range(self.dim))
        self.axis0 = lambda row, col: tuple(row * pow(self.dim, 2) + col + i * self.dim for i in range(self.dim))
        self.axis2 = lambda row, col: tuple(row + col * self.dim + i * pow(self.dim, 2) for i in range(self.dim))
        self.add_constraints(self.pics[0], self.axis0)
        self.add_constraints(self.pics[1], self.axis1)
        self.add_constraints(self.pics[2], self.axis2)

    def add_constraints(self, pic, add_row):
        for row in range(self.dim):
            for col in range(self.dim):
                if pic[col, row]:
                    self.problem.addConstraint(has, add_row(row, col))
                else:
                    self.problem.addConstraint(hasnt, add_row(row, col))

    def solve(self):
        solution = self.problem.getSolution()
        if solution is None:
            return None
        return np.array([solution[i] for i in range(pow(self.dim, 3))]).reshape((self.dim, self.dim, self.dim))


if __name__ == '__main__':
    solver = CSPSolver([pic1, pic2, pic3])
    print(solver.solve())
    # print_3d_matrix(solver.solve())
