"""ORSolver - Comprehensive Operations Research Solver.

Intégré au module d'optimisation de Polytech :
- LP, Dual Simplex, Duality, Sensitivity, Goal Programming
- MIP, Branch & Bound, Gomory Cuts, Knapsack 0-1
- Transport, Affectation (Hongrois)
- Réseaux (Dijkstra, Bellman-Ford, Floyd-Warshall, A*, Flot Max, Min Cost Flow, Kruskal, Prim, PERT/CPM)
- Théorie des jeux (Minimax von Neumann, élagage Alpha-Beta)
- Processus de Markov (MDP)
- Aide à la décision multicritère (AHP de Saaty, TOPSIS)
- Décision dans l'incertain (Wald, Maximax, Hurwicz, Savage, Laplace, EVPI)
- Fiabilité et maintenance (Systèmes série/parallèle/k-sur-n, Weibull, MTBF/MTTF)
- Programmation dynamique (Multiplication de matrices, Rendu de monnaie)
- Simulation (Monte Carlo pour PERT)
- Files d'attente (M/M/1, M/M/s Erlang-C)
- Gestion des stocks (Wilson EOQ, pénuries, stock de sécurité)
- Métaheuristiques (TSP: 2-Opt, Recuit Simulé)
- Optimisation non linéaire (Section dorée 1D, gradient Armijo)
- Parser CPLEX .lp
- Visualiseur 2D dynamique
"""

from __future__ import annotations

from polytech.optimisation.orsolver import (
    cli,
    core,
    decision,
    dynamic_programming,
    game_theory,
    heuristics,
    inventory,
    legacy,
    lp,
    mcda,
    mdp,
    mip,
    network,
    nlp,
    queuing,
    reliability,
    simulation,
    transport,
    visualizer,
)
from polytech.optimisation.orsolver.core.model import (
    Constraint,
    Problem,
    Sense,
    Variable,
)
from polytech.optimisation.orsolver.core.solution import Solution, SolutionStatus
from polytech.optimisation.orsolver.legacy import GraphicalORSolver, SimplexSolver
from polytech.optimisation.orsolver.lp.simplex import TwoPhaseSimplexSolver

__version__ = "1.0.0"

__all__ = [
    "Constraint",
    "GraphicalORSolver",
    "Problem",
    "Sense",
    "SimplexSolver",
    "Solution",
    "SolutionStatus",
    "TwoPhaseSimplexSolver",
    "Variable",
    "cli",
    "core",
    "decision",
    "dynamic_programming",
    "game_theory",
    "heuristics",
    "inventory",
    "legacy",
    "lp",
    "mcda",
    "mdp",
    "mip",
    "network",
    "nlp",
    "queuing",
    "reliability",
    "simulation",
    "transport",
    "visualizer",
]
