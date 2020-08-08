from csp_solvers import *



def csp_solver_factory(pics, solver='MRV'):
    if solver == 'MRV':
        return MRV(pics)
    else:
        return CSPSolver(pics, solver)
