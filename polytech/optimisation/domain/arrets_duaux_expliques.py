"""Les deux arrets de l'algorithme dual, chacun rapporte au cas du V.1 qui le fonde.

1. La base, duale admissible, est aussi primale admissible : optimale.
2.a Une variable de base negative dont la ligne n'a aucun pivot negatif :
    le parametre theta croit sans limite, le dual n'est pas borne, et le
    primal est impossible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.raisonnement.domain.etapes import Etape, Raisonnement

if TYPE_CHECKING:
    import sympy as sp

    from polytech.optimisation.domain.tableau import TableauSimplexe

SOURCE: Final = "Optimisation/SLIDES/Slides_05_Lalgorithme_dual_simplexe.pdf"
QUESTION: Final = "Que fait une iteration de l'algorithme dual sur ce tableau ?"


def _liste(valeurs: sp.Matrix) -> str:
    return ", ".join(str(v) for v in list(valeurs))


def expliquer_l_arret_optimal(tableau: TableauSimplexe) -> Raisonnement:
    """V.1, cas 1 : duale et primale admissible, donc optimale."""
    etape = Etape(
        "la solution de base est optimale",
        f"la base est duale admissible et les a_i0 "
        f"({_liste(tableau.coefficients[1:, 0])}) sont tous non negatifs : elle est "
        "aussi primale admissible",
        f"{SOURCE}, V.1, cas 1, p. 114",
    )
    return Raisonnement(QUESTION, (etape,), f"solution optimale : z = {tableau.valeur}")


def expliquer_l_arret_impossible(tableau: TableauSimplexe, indice: int) -> Raisonnement:
    """V.1, cas 2.a : une variable negative que rien ne peut remplacer."""
    ligne = tableau.base.index(indice) + 1
    etape = Etape(
        f"x_{indice} = {tableau.coefficients[ligne, 0]} est negative et ne peut "
        "pas sortir",
        f"les a_{indice}j ({_liste(tableau.coefficients[ligne, 1:])}) sont tous non "
        "negatifs : aucun pivot negatif, theta croit sans limite",
        f"{SOURCE}, V.1, cas 2.a, p. 115",
    )
    conclusion = "probleme impossible : le dual n'est pas borne"
    return Raisonnement(QUESTION, (etape,), conclusion)
