"""Chapitre I.2.2 : les hypotheses du modele lineaire, et ce qui en sort.

Trois hypotheses fondent le modele (page 31) :

* l'**additivite** -- les activites n'interagissent pas, ni dans la
  fonction economique ni dans les contraintes ;
* la **proportionnalite** -- gain et utilisation sont proportionnels au
  niveau de l'activite, donc du premier degre ;
* la **divisibilite** -- les niveaux se mesurent par des fractions.

Les deux premieres se lisent sur une expression : un terme
`x_i^k_i x_j^k_j` de degre total au moins deux les viole. La troisieme
porte sur la nature des variables, non sur une formule ; l'abandonner
mene aux variables entieres du chapitre XIV.

La linearite rend le **gradient** de la fonction economique constant.
Deux ecarts a la proportionnalite restent traitables (pages 31-32) : un
cout unitaire **lineaire par morceaux**, et un **cout fixe**, paye des
que l'activite tourne, que l'on modelise par une variable binaire.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence

    from polytech.optimisation.domain.formulation import ProgrammeGeneral


class Hypothese(Enum):
    """Les hypotheses du modele qu'une expression peut violer."""

    ADDITIVITE = "additivite"
    PROPORTIONNALITE = "proportionnalite"


def hypotheses_violees(
    expression: sp.Expr,
    variables: Sequence[sp.Symbol],
) -> frozenset[Hypothese]:
    """Les hypotheses que cette fonction des activites ne respecte pas."""
    try:
        polynome = sp.Poly(expression, *variables)
    except sp.PolynomialError:
        # Ni somme ni puissance entiere : `sqrt(x)` n'est pas proportionnel a `x`.
        return frozenset({Hypothese.PROPORTIONNALITE})
    violees: set[Hypothese] = set()
    for exposants in polynome.monoms():
        if sum(1 for e in exposants if e > 0) > 1:
            violees.add(Hypothese.ADDITIVITE)
        if any(e > 1 for e in exposants):
            violees.add(Hypothese.PROPORTIONNALITE)
    return frozenset(violees)


def gradient(programme: ProgrammeGeneral) -> tuple[sp.Expr, ...]:
    """`dz/dx_j` pour chaque variable : constant, et egal a `c_j`."""
    variables = sp.symbols(f"x_1:{len(programme.objectif) + 1}")
    economique = programme.valeur(variables)
    return tuple(sp.diff(economique, x) for x in variables)


def cout_unitaire_par_morceaux(
    niveau: object,
    seuils: Sequence[object],
    couts: Sequence[object],
) -> object:
    """`c^(k)` tant que `a_(k-1) < x <= a_k`, et le dernier cout au-dela."""
    if len(couts) != len(seuils) + 1:
        msg = f"un cout de plus que de seuils : {len(couts)} pour {len(seuils)}"
        raise ValueError(msg)
    for seuil, cout in zip(seuils, couts, strict=False):
        if sp.sympify(niveau) <= sp.sympify(seuil):
            return cout
    return couts[-1]


def variable_de_charge(niveau: object) -> int:
    """La variable binaire `y` : `0` si `x = 0`, `1` si `x > 0`."""
    if sp.sympify(niveau) < 0:
        msg = f"niveau d'activite negatif : {niveau}"
        raise ValueError(msg)
    return 0 if sp.sympify(niveau) == 0 else 1


def cout_a_charge_fixe(niveau: object, fixe: object, proportionnel: object) -> sp.Expr:
    """`c = a y + b x` : rien si l'activite est a l'arret, `a + b x` sinon."""
    charge = variable_de_charge(niveau)
    return sp.sympify(fixe) * charge + sp.sympify(proportionnel) * sp.sympify(niveau)
