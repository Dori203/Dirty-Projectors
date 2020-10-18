import argparse
import GUI
import image_processing
import factories
import genetic3rd

_solvers_choices = ['mrv', 'mc', 'genetic', 'greedy', 'hill', 'local', 'annealing']
_init_state_choices = ['blank', 'black', 'random']
_fitness_func_choices = ['l1', 'l2', 'punish_empty', 'punish_black']
_successor_func_choices = ['naive', 'onion', 'flip', 'neighbouring', 'column']
_path_help = "Path to folder with 1,2,3 or 4 pictures to run on (csp supports up to 3 pictures)."
_dim_help = "Dimension of the pictures (default = 40)."
_iterations_help = "Number of iterations for search algorithms or generations in the genetic search (default = 5000)."
_solver_help = "Type of algorithm to use (default = local greedy search)."
_init_help = "Initial state for local search algorithms (default = blank)."
_fitness_help = "Loss function to use in local search (default = penalize black pixels)."
_successor_help = "Successor function for local search (default = neighbouring)."


def get_solver(args):
    solver_name = args.solver
    if solver_name in ['mrv', 'mc']:
        return factories.csp_solver_factory(args.path, (args.dim, args.dim), solver_name)
    elif solver_name in ['greedy', 'hill', 'local', 'annealing', 'genetic']:
        processor = image_processing.ImageProcessor(args.path, image_processing.GRAY, (args.dim, args.dim))
        fitness_func = factories.fitness_func_factory(args.fitness, processor)
        if solver_name in ['genetic']:
            params = {'num_generations': args.iterations, 'population_size': 100, 'mutation_probability': 0.4,
                      'crossover_probability': 0.4}
            return genetic3rd.GeneticSolver((args.dim, args.dim, args.dim), fitness_func, args.fitness, params)
        successor_func = factories.successor_func_factory(args.successor)
        init_func = factories.init_func_factory(args.init_state, args.dim)
        return factories.heuristic_solver_factory(solver_name)(fitness_func, args.fitness, successor_func, init_func,
                                                               args.iterations)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p', type=str, required=True, help=_path_help)
    parser.add_argument('--dim', '-d', type=int, default=40, help=_dim_help)
    parser.add_argument('--iterations', '-i', type=int, default=5000, help=_iterations_help)
    parser.add_argument('--solver', '-s', choices=_solvers_choices, default='greedy', help=_solver_help)
    parser.add_argument('--init_state', '-is', choices=_init_state_choices, default='blank', help=_init_help)
    parser.add_argument('--fitness', '-f', choices=_fitness_func_choices, default='punish_empty', help=_fitness_help)
    parser.add_argument('--successor', '-sf', choices=_successor_func_choices, default='neighbouring',
                        help=_successor_help)
    arguments = parser.parse_args()
    solver = get_solver(arguments)
    solver.solve()
    if arguments.solver not in ["genetic"]:
        GUI.show_result(solver, arguments.solver)
    else:
        solver.get_answer()
    input("Press any key to continue...")
