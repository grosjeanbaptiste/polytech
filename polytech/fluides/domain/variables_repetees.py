"""La methode des variables repetees, qui fabrique les groupes sans dimension.

Syllabus du theoreme de Vaschy-Buckingham (Cengel, section 7-4). Le
theoreme dit combien de groupes attendre ; la methode les construit.
Chaque parametre restant est multiplie par les parametres repetes eleves
a des exposants inconnus, et l'on force l'exposant de chaque dimension
primaire a zero : un systeme lineaire, a resoudre exactement.

Les regles de choix de la table 7-3 sont celles qui rendent ce systeme
soluble d'une seule facon : les parametres repetes ne forment pas un
groupe a eux seuls, et portent toutes les dimensions du probleme. Un
groupe obtenu se manipule sans cesser d'etre sans dimension (table 7-4) :
on l'eleve a une puissance, on le multiplie par un autre.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

type Dimensions = Mapping[str, int]
"""L'exposant de chaque dimension primaire d'un parametre."""
type Groupe = dict[str, sp.Expr]
"""L'exposant de chaque parametre d'un produit."""


def dimensions_du_produit(
    groupe: Mapping[str, sp.Expr | int], parametres: Mapping[str, Dimensions]
) -> dict[str, sp.Expr]:
    """Les dimensions du produit ; vide s'il est sans dimension."""
    total: dict[str, sp.Expr] = {}
    for nom, exposant in groupe.items():
        if nom not in parametres:
            msg = f"parametre inconnu : {nom}"
            raise ValueError(msg)
        for dimension, puissance in parametres[nom].items():
            total[dimension] = total.get(dimension, sp.S.Zero) + exposant * puissance
    return {dimension: e for dimension, e in total.items() if e != 0}


def groupe_pi(
    parametre: str, repetes: Sequence[str], parametres: Mapping[str, Dimensions]
) -> Groupe:
    """Le groupe du parametre : lui-meme, fois les repetes a leurs exposants."""
    exposants = sp.symbols(f"a0:{len(repetes)}")
    inconnus = dict(zip(repetes, exposants, strict=True))
    essai = {parametre: sp.S.One, **inconnus}
    seuls = dimensions_du_produit(inconnus, parametres)
    libres = sp.linsolve(list(seuls.values()), exposants)
    if libres != sp.FiniteSet((0,) * len(repetes)):
        msg = f"les parametres repetes {tuple(repetes)} forment un groupe a eux seuls"
        raise ValueError(msg)
    equations = list(dimensions_du_produit(essai, parametres).values())
    solutions = sp.linsolve(equations, exposants)
    if not solutions:
        msg = f"les repetes {tuple(repetes)} ne portent pas toutes les dimensions"
        raise ValueError(msg)
    (solution,) = solutions
    groupe = {parametre: sp.S.One, **dict(zip(repetes, solution, strict=True))}
    return {nom: exposant for nom, exposant in groupe.items() if exposant != 0}


def groupes_pi(
    dependante: str, repetes: Sequence[str], parametres: Mapping[str, Dimensions]
) -> tuple[Groupe, ...]:
    """Tous les groupes, celui de la variable dependante en tete."""
    if dependante in repetes:
        msg = f"la variable dependante {dependante} ne peut pas etre repetee"
        raise ValueError(msg)
    restants = [nom for nom in parametres if nom not in (dependante, *repetes)]
    return tuple(groupe_pi(nom, repetes, parametres) for nom in (dependante, *restants))


def elever(groupe: Mapping[str, sp.Expr], exposant: sp.Expr | int) -> Groupe:
    """Le groupe eleve a une puissance : il reste sans dimension."""
    return {nom: valeur * exposant for nom, valeur in groupe.items()}


def multiplier(premier: Mapping[str, sp.Expr], second: Mapping[str, sp.Expr]) -> Groupe:
    """Le produit de deux groupes, les exposants nuls retires."""
    somme = dict(premier)
    for nom, valeur in second.items():
        somme[nom] = somme.get(nom, sp.S.Zero) + valeur
    return {nom: valeur for nom, valeur in somme.items() if valeur != 0}
