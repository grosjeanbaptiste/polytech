from polytech.optimisation.heuristics.tsp import (
    compute_tour_cost,
    nearest_neighbor_tsp,
    simulated_annealing_tsp,
    two_opt_tsp,
)

tsp_2opt = two_opt_tsp
tsp_simulated_annealing = simulated_annealing_tsp

__all__ = [
    "compute_tour_cost",
    "nearest_neighbor_tsp",
    "simulated_annealing_tsp",
    "tsp_2opt",
    "tsp_simulated_annealing",
    "two_opt_tsp",
]
