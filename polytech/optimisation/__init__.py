"""Contexte borné : optimisation mathématique et recherche opérationnelle.

Solveurs de recherche opérationnelle et d'aide à la décision :
- Programmation linéaire (simplexe 2-phases, dual, sensibilité, goal programming)
- Programmation en nombres entiers (branch-and-bound, Gomory, sac à dos 0-1)
- Problèmes de transport (VAM, MODI, coin nord-ouest) et affectation (Hongrois)
- Optimisation de réseaux (Dijkstra, Bellman-Ford, Floyd, A*, flots, MST, PERT)
- Théorie des jeux (Minimax à somme nulle, élagage alpha-bêta)
- Processus décisionnels de Markov (MDP, Bellman)
- Aide à la décision multicritère (AHP de Saaty, TOPSIS)
- Théorie de la décision dans l'incertain (Wald, Savage, Hurwicz, Laplace, EVPI)
- Fiabilité industrielle et maintenance (systèmes k-sur-n, Weibull, MTBF/MTTF)
- Programmation dynamique (chaîne de matrices, rendu de monnaie)
- Simulation de risque (Monte Carlo pour PERT stochastique)
- Théorie des files d'attente (M/M/1, M/M/s Erlang-C)
- Gestion des stocks (modèle de Wilson EOQ, pénuries, point de commande)
- Métaheuristiques pour le TSP (plus proche voisin, 2-Opt, recuit simulé)
- Optimisation non-linéaire (section dorée 1D, gradient avec recherche d'Armijo)
- Parser standard CPLEX .lp et méthode graphique 2D
"""

from __future__ import annotations

from polytech.optimisation import (
    application,
    cli,
    core,
    decision,
    domain,
    dynamic_programming,
    game_theory,
    heuristics,
    infrastructure,
    inventory,
    legacy,
    lp,
    mcda,
    mdp,
    mip,
    network,
    nlp,
    orsolver,
    queuing,
    reliability,
    simulation,
    transport,
    visualizer,
)
from polytech.optimisation.core.model import (
    Constraint,
    Problem,
    Sense,
    Variable,
)
from polytech.optimisation.core.solution import Solution, SolutionStatus
from polytech.optimisation.legacy import GraphicalORSolver, SimplexSolver
from polytech.optimisation.lp.simplex import TwoPhaseSimplexSolver

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
    "application",
    "cli",
    "core",
    "decision",
    "domain",
    "dynamic_programming",
    "game_theory",
    "heuristics",
    "infrastructure",
    "inventory",
    "legacy",
    "lp",
    "mcda",
    "mdp",
    "mip",
    "network",
    "nlp",
    "orsolver",
    "queuing",
    "reliability",
    "simulation",
    "transport",
    "visualizer",
]
