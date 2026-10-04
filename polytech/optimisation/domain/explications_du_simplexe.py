"""Une iteration du simplexe, dite comme le cours la dit a cote de ses tableaux.

`algorithme.phase_deux` rend les tableaux ; ce module rend ce que le
cours ecrit en regard, page 57 : quelle variable entre et **pourquoi**
(son coefficient), laquelle sort et **pourquoi** (le plus petit rapport),
quel est le pivot. Et, aux deux arrets, la propriete qui les justifie.

Chaque etape porte la reference de la propriete du chapitre II qui la
fonde : un corrige peut la citer telle quelle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from polytech.optimisation.domain.algorithme import phase_deux
from polytech.optimisation.domain.contraintes import Optimisation
from polytech.optimisation.domain.criteres import (
    colonne_sans_solution_finie,
    est_optimal,
    variable_entrante,
)
from polytech.optimisation.domain.interpretation import gain_de_l_iteration
from polytech.optimisation.domain.regles import sortante_lexicographique
from polytech.raisonnement.domain.etapes import Etape, Raisonnement

if TYPE_CHECKING:
    import sympy as sp

    from polytech.optimisation.domain.tableau import TableauSimplexe

SOURCE: Final = "Optimisation/SLIDES/Slides_02_Lalgorithme_simplexe.pdf"
QUESTION: Final = "Que fait une iteration du simplexe sur ce tableau ?"


def _liste(valeurs: sp.Matrix) -> str:
    return ", ".join(str(v) for v in list(valeurs))


def _arret_optimal(tableau: TableauSimplexe, sens: Optimisation) -> Raisonnement:
    signe = "non positifs" if sens is Optimisation.MIN else "non negatifs"
    coefficients = _liste(tableau.coefficients[0, 1:])
    etape = Etape(
        "la solution de base est optimale",
        f"les coefficients a_0j ({coefficients}) sont tous {signe}",
        f"{SOURCE}, propriete II.2.3, p. 52",
    )
    return Raisonnement(QUESTION, (etape,), f"solution optimale : z = {tableau.valeur}")


def _arret_non_borne(tableau: TableauSimplexe, colonne: int) -> Raisonnement:
    rang = tableau.hors_base.index(colonne) + 1
    etape = Etape(
        f"x_{colonne} peut entrer, mais rien ne borne son augmentation",
        f"ses coefficients a_i{colonne} ({_liste(tableau.coefficients[1:, rang])}) "
        "sont tous negatifs ou nuls : aucun pivot positif",
        f"{SOURCE}, propriete II.2.1, p. 50",
    )
    return Raisonnement(QUESTION, (etape,), "pas de solution optimale finie")


def _changement(tableau: TableauSimplexe, sens: Optimisation) -> Raisonnement:
    entrante = variable_entrante(tableau, sens)
    variables = max((*tableau.base, *tableau.hors_base))
    sortante = sortante_lexicographique(tableau, entrante, variables)
    colonne = tableau.hors_base.index(entrante) + 1
    ligne = tableau.base.index(sortante) + 1
    extremum, nature = (
        ("max", "minimisation") if sens is Optimisation.MIN else ("min", "maximisation")
    )
    rapports = [
        f"{tableau.coefficients[i, 0]}/{tableau.coefficients[i, colonne]}"
        for i in range(1, tableau.coefficients.rows)
        if tableau.coefficients[i, colonne] > 0
    ]
    pivot = tableau.coefficients[ligne, colonne]
    choisi = f"{tableau.coefficients[ligne, 0]}/{pivot}"
    etapes = (
        Etape(
            f"x_{entrante} entre dans la base",
            f"{tableau.coefficients[0, colonne]} = {extremum}("
            f"{_liste(tableau.coefficients[0, 1:])}) (probleme de {nature})",
            f"{SOURCE}, II.4.2, p. 55",
        ),
        Etape(
            f"x_{sortante} sort de la base",
            f"{choisi} = min({', '.join(rapports)}) sur les a_ik positifs",
            f"{SOURCE}, propriete II.2.2, p. 50",
        ),
        Etape(
            f"le pivot est a_{sortante}{entrante} = {pivot}",
            "c'est le coefficient a l'intersection de la ligne sortante et de la "
            "colonne entrante",
            f"{SOURCE}, II.3, p. 54",
        ),
    )
    nouvelle = tableau.valeur + gain_de_l_iteration(tableau, entrante)
    return Raisonnement(QUESTION, etapes, f"z passe de {tableau.valeur} a {nouvelle}")


def expliquer_une_iteration(
    tableau: TableauSimplexe, sens: Optimisation
) -> Raisonnement:
    """Ce que l'algorithme fait de ce tableau, et la propriete qui le justifie."""
    if est_optimal(tableau, sens):
        return _arret_optimal(tableau, sens)
    colonne = colonne_sans_solution_finie(tableau, sens)
    if colonne is not None:
        return _arret_non_borne(tableau, colonne)
    return _changement(tableau, sens)


def expliquer_la_resolution(
    depart: TableauSimplexe, sens: Optimisation
) -> Raisonnement:
    """Toutes les iterations enchainees, du tableau de depart a l'arret."""
    resolution = phase_deux(depart, sens)
    raisonnement = expliquer_une_iteration(resolution.tableaux[0], sens)
    for tableau in resolution.tableaux[1:]:
        raisonnement = raisonnement.puis(expliquer_une_iteration(tableau, sens))
    return raisonnement
