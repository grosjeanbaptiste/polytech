"""Contexte borné : optimisation mathématique et recherche opérationnelle.

Programmation linéaire (simplexe, grand M, dual simplexe), dualité,
variables entières (branch-and-bound), et solveurs de recherche opérationnelle
(ORSolver : graphes, théorie des jeux, files d'attente, stocks, MCDA, fiabilité).
"""

from __future__ import annotations

from polytech.optimisation import application, domain, infrastructure, orsolver
from polytech.optimisation.orsolver import (
    decision,
    dynamic_programming,
    game_theory,
    heuristics,
    inventory,
    mcda,
    mdp,
    network,
    nlp,
    queuing,
    reliability,
    simulation,
)

__all__ = [
    "application",
    "decision",
    "domain",
    "dynamic_programming",
    "game_theory",
    "heuristics",
    "infrastructure",
    "inventory",
    "mcda",
    "mdp",
    "network",
    "nlp",
    "orsolver",
    "queuing",
    "reliability",
    "simulation",
]
