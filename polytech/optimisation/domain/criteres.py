"""Chapitre II.2 : les trois proprietes qui fondent une iteration du simplexe.

Sur le tableau d'une solution de base admissible, et pour une
**minimisation** :

* II.2.1 -- s'il existe `k` hors base avec `a_0k > 0` et `a_ik <= 0` pour
  tout `i`, la fonction economique n'est pas bornee : pas de solution
  optimale finie ;
* II.2.2 -- sinon, faire entrer un `x_k` tel que `a_0k > 0` et sortir le
  `x_l` qui realise `min(a_i0 / a_ik)` sur les `a_ik > 0` donne une
  solution de base admissible au moins aussi bonne ;
* II.2.3 -- si `a_0j <= 0` pour tout `j`, la solution est optimale.

Pour une **maximisation**, le raisonnement est le meme en inversant le
signe des `a_0j` (remarque 1, page 56). La variable entrante est celle de
la plus forte variation unitaire : `max(a_0j)` en minimisation.

Un `a_0j` peut porter la constante `M` de la phase I ; il se compare
alors par son terme en `M` d'abord (`grand_m.ordre`).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.optimisation.domain.contraintes import Optimisation
from polytech.optimisation.domain.grand_m import ordre

if TYPE_CHECKING:
    import sympy as sp

    from polytech.optimisation.domain.tableau import TableauSimplexe


def _attraits(tableau: TableauSimplexe, sens: Optimisation) -> list[sp.Expr]:
    """Les `a_0j`, signes pour qu'un attrait positif ameliore `z`."""
    signe = 1 if sens is Optimisation.MIN else -1
    return [signe * a for a in tableau.coefficients[0, 1:]]


def _ameliore(attrait: sp.Expr) -> bool:
    """Vrai si l'attrait est positif, la constante `M` etant aussi grande qu'on veut."""
    return ordre(attrait) > (0, 0)


def est_optimal(tableau: TableauSimplexe, sens: Optimisation) -> bool:
    """Propriete II.2.3 : aucune variable hors base ne peut ameliorer `z`."""
    return not any(_ameliore(attrait) for attrait in _attraits(tableau, sens))


def variable_entrante(tableau: TableauSimplexe, sens: Optimisation) -> int:
    """L'indice `k` de la plus forte amelioration unitaire de `z`."""
    attraits = _attraits(tableau, sens)
    meilleur = max(attraits, key=ordre)
    if not _ameliore(meilleur):
        msg = f"aucune variable entrante : le tableau est optimal ({tableau.base})"
        raise ValueError(msg)
    return tableau.hors_base[attraits.index(meilleur)]


def variables_sortantes(tableau: TableauSimplexe, entrante: int) -> tuple[int, ...]:
    """Les indices `l` qui realisent `min(a_i0 / a_ik)` sur les `a_ik > 0`.

    Plusieurs indices a egalite annoncent une solution degeneree (II.6.2).
    """
    colonne = tableau.hors_base.index(entrante) + 1
    rapports = {
        indice: tableau.coefficients[ligne, 0] / tableau.coefficients[ligne, colonne]
        for ligne, indice in enumerate(tableau.base, start=1)
        if tableau.coefficients[ligne, colonne] > 0
    }
    if not rapports:
        return ()
    return tuple(
        i for i, rapport in rapports.items() if rapport == min(rapports.values())
    )


def colonne_sans_solution_finie(
    tableau: TableauSimplexe,
    sens: Optimisation,
) -> int | None:
    """Propriete II.2.1 : un `k` attrayant sans aucun pivot positif, ou `None`."""
    for indice, attrait in zip(
        tableau.hors_base, _attraits(tableau, sens), strict=True
    ):
        if _ameliore(attrait) and not variables_sortantes(tableau, indice):
            return indice
    return None


def colonnes_reduisant_la_penalite(
    tableau: TableauSimplexe,
    sens: Optimisation,
) -> tuple[int, ...]:
    """Les variables hors base dont le terme en `M` ameliore `z`.

    Ce sont elles, et elles seules, qui peuvent encore faire decroitre les
    variables artificielles : tant qu'il en reste, la phase I n'est pas
    finie.
    """
    return tuple(
        indice
        for indice, attrait in zip(
            tableau.hors_base, _attraits(tableau, sens), strict=True
        )
        if ordre(attrait)[0] > 0
    )
