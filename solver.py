"""general interface for solving the problem - each successor implementing another
concept"""


class Solver:

    def solve(self):
        pass

    def get_answer(self):
        pass

    def get_answer_loss(self):
        pass
