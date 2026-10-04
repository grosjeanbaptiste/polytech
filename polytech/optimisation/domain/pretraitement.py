"""Chapitre XIV.3.5 c) : le pretraitement d'un programme en variables entieres.

Avant toute separation, les logiciels simplifient ou renforcent la
formulation en confrontant chaque contrainte `somme(a_j x_j) <= b` aux
bornes `l_j <= x_j <= u_j` de ses variables entieres. Le premier membre
est **minimal** quand les variables de coefficient positif sont a leur
borne inferieure et les autres a leur borne superieure ; **maximal** dans
le cas oppose. Alors :

* maximum `<= b` : la contrainte est **redondante**, on la supprime ;
* minimum `> b` : elle est **impossible**, et le probleme avec elle ;
* sinon chaque variable recoit une borne plus etroite, le reste du
  premier membre etant pris a son minimum -- et arrondie, la variable
  etant entiere ;
* une variable dont les deux bornes se rejoignent est **fixee** et sort
  de la contrainte.

Aucune solution entiere n'est perdue : seule la relaxation se resserre.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.optimisation.domain.contraintes import Contrainte, Sens

if TYPE_CHECKING:
    from collections.abc import Sequence

type Bornes = tuple[tuple[int, int], ...]


def _inferieure(
    contrainte: Contrainte, bornes: Sequence[tuple[int, int]]
) -> Contrainte:
    """La contrainte ecrite `<=`, ses bornes verifiees."""
    if len(bornes) != len(contrainte.coefficients):
        msg = f"{len(bornes)} bornes pour {len(contrainte.coefficients)} variables"
        raise ValueError(msg)
    if contrainte.sens is Sens.EGAL:
        msg = "une egalite ne se pretraite pas ainsi : la dedoubler en deux inegalites"
        raise ValueError(msg)
    return contrainte if contrainte.sens is Sens.INFERIEUR else contrainte.retournee()


def _termes(
    contrainte: Contrainte, bornes: Sequence[tuple[int, int]]
) -> list[tuple[sp.Expr, sp.Expr]]:
    """Pour chaque variable, le plus petit et le plus grand `a_j x_j`."""
    termes = []
    for a, (bas, haut) in zip(contrainte.coefficients, bornes, strict=True):
        extremes = (a * bas, a * haut)
        termes.append((sp.Min(*extremes), sp.Max(*extremes)))
    return termes


def etendue_du_premier_membre(
    contrainte: Contrainte,
    bornes: Sequence[tuple[int, int]],
) -> tuple[sp.Expr, sp.Expr]:
    """Le minimum et le maximum du premier membre sur la boite des bornes."""
    termes = _termes(contrainte, bornes)
    return sp.Add(*(t[0] for t in termes)), sp.Add(*(t[1] for t in termes))


def est_redondante(contrainte: Contrainte, bornes: Sequence[tuple[int, int]]) -> bool:
    """Vrai si les bornes suffisent a la verifier."""
    ecrite = _inferieure(contrainte, bornes)
    return bool(etendue_du_premier_membre(ecrite, bornes)[1] <= ecrite.second_membre)


def est_impossible(contrainte: Contrainte, bornes: Sequence[tuple[int, int]]) -> bool:
    """Vrai si aucun point de la boite ne la verifie."""
    ecrite = _inferieure(contrainte, bornes)
    return bool(etendue_du_premier_membre(ecrite, bornes)[0] > ecrite.second_membre)


def resserrer_les_bornes(
    contrainte: Contrainte,
    bornes: Sequence[tuple[int, int]],
) -> Bornes:
    """Les bornes que la contrainte impose a chaque variable entiere."""
    ecrite = _inferieure(contrainte, bornes)
    termes = _termes(ecrite, bornes)
    minimum = sp.Add(*(t[0] for t in termes))
    resserrees = []
    for a, (bas, haut), (plus_petit, _) in zip(
        ecrite.coefficients, bornes, termes, strict=True
    ):
        reste = ecrite.second_membre - (minimum - plus_petit)
        plafond = min(haut, int(sp.floor(reste / a))) if a > 0 else haut
        plancher = max(bas, int(sp.ceiling(reste / a))) if a < 0 else bas
        resserrees.append((plancher, plafond))
    return tuple(resserrees)


def fixer_les_variables(
    contrainte: Contrainte,
    bornes: Sequence[tuple[int, int]],
) -> Contrainte:
    """La contrainte, les variables aux bornes confondues passees au second membre."""
    fixees = [bas == haut for bas, haut in bornes]
    constante = sp.Add(
        *(
            a * bas
            for a, (bas, _), fixee in zip(
                contrainte.coefficients, bornes, fixees, strict=True
            )
            if fixee
        )
    )
    coefficients = tuple(
        0 if fixee else a
        for a, fixee in zip(contrainte.coefficients, fixees, strict=True)
    )
    return Contrainte(
        coefficients, contrainte.sens, contrainte.second_membre - constante
    )
