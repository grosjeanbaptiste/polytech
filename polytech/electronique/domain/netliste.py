"""L'analyse nodale modifiee, assemblee a partir d'une liste d'elements.

Chaque element depose sa contribution dans la matrice -- c'est ce que le
cours fait a la main, noeud par noeud. Convention du TP 4 : la loi des
noeuds compte les courants **entrants** positivement, et les termes
constants passent au second membre. Le noeud 0 est la masse : il n'a pas
d'equation.

Les inconnues sont rangees dans l'ordre des enonces : les tensions des
noeuds 1 a n, puis les courants des sources de tension, dans l'ordre ou
elles sont declarees.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp
from sympy.polys.matrices import DomainMatrix

if TYPE_CHECKING:
    from collections.abc import Sequence


@dataclass(frozen=True)
class Admittance:
    """Une branche passive entre deux noeuds."""

    noeud_a: int
    noeud_b: int
    valeur: sp.Expr


@dataclass(frozen=True)
class SourceDeCourant:
    """Un courant qui traverse la source de `depart` vers `arrivee`."""

    depart: int
    arrivee: int
    valeur: sp.Expr


@dataclass(frozen=True)
class SourceCommandee:
    """Le courant `g (V_plus - V_moins)`, de `depart` vers `arrivee`."""

    depart: int
    arrivee: int
    transconductance: sp.Expr
    plus: int
    moins: int


@dataclass(frozen=True)
class SourceDeTension:
    """`V_plus - V_moins = valeur` ; `courant` sort par la borne plus."""

    plus: int
    moins: int
    valeur: sp.Expr
    courant: sp.Symbol


@dataclass(frozen=True)
class SourceDeTensionCommandee:
    """`V_plus - V_moins = gain (V_a - V_b)`, ou `commande = (a, b)`."""

    plus: int
    moins: int
    gain: sp.Expr
    commande: tuple[int, int]
    courant: sp.Symbol


SourcesDeTension = SourceDeTension | SourceDeTensionCommandee
Element = Admittance | SourceDeCourant | SourceCommandee | SourcesDeTension


def _noeuds(element: Element) -> tuple[int, ...]:
    if isinstance(element, Admittance):
        return element.noeud_a, element.noeud_b
    if isinstance(element, SourceDeCourant):
        return element.depart, element.arrivee
    if isinstance(element, SourceCommandee):
        return element.depart, element.arrivee, element.plus, element.moins
    if isinstance(element, SourceDeTensionCommandee):
        return element.plus, element.moins, *element.commande
    return element.plus, element.moins


def _ajouter(matrice: sp.Matrix, ligne: int, colonne: int, valeur: sp.Expr) -> None:
    """Depose un terme, sauf sur la masse, qui n'a ni ligne ni inconnue."""
    if ligne > 0 and colonne > 0:
        matrice[ligne - 1, colonne - 1] += valeur


def _deposer(matrice: sp.Matrix, membre: sp.Matrix, element: Element) -> None:
    if isinstance(element, Admittance):
        a, b, y = element.noeud_a, element.noeud_b, element.valeur
        for ici, la in ((a, b), (b, a)):
            _ajouter(matrice, ici, ici, -y)
            _ajouter(matrice, ici, la, y)
    elif isinstance(element, SourceDeCourant):
        for noeud, signe in ((element.arrivee, -1), (element.depart, 1)):
            if noeud > 0:
                membre[noeud - 1] += signe * element.valeur
    elif isinstance(element, SourceCommandee):
        g = element.transconductance
        for noeud, signe in ((element.arrivee, 1), (element.depart, -1)):
            _ajouter(matrice, noeud, element.plus, signe * g)
            _ajouter(matrice, noeud, element.moins, -signe * g)


def systeme_nodal_modifie(
    elements: Sequence[Element],
    noeuds: int,
) -> tuple[sp.Matrix, sp.Matrix]:
    """La matrice du systeme et son second membre."""
    for element in elements:
        hors = [n for n in _noeuds(element) if not 0 <= n <= noeuds]
        if hors:
            msg = f"noeud {hors[0]} hors du circuit a {noeuds} noeuds : {element}"
            raise ValueError(msg)
    sources = [e for e in elements if isinstance(e, SourcesDeTension)]
    taille = noeuds + len(sources)
    matrice, membre = sp.zeros(taille, taille), sp.zeros(taille, 1)
    for element in elements:
        _deposer(matrice, membre, element)
    for rang, source in enumerate(sources, start=noeuds + 1):
        for noeud, signe in ((source.plus, 1), (source.moins, -1)):
            _ajouter(matrice, noeud, rang, signe)
            _ajouter(matrice, rang, noeud, signe)
        if isinstance(source, SourceDeTension):
            membre[rang - 1] = source.valeur
            continue
        for noeud, signe in zip(source.commande, (-1, 1), strict=True):
            _ajouter(matrice, rang, noeud, signe * source.gain)
    return matrice, membre


def resoudre(elements: Sequence[Element], noeuds: int) -> sp.Matrix:
    """Les tensions des noeuds, puis les courants des sources de tension."""
    matrice, membre = systeme_nodal_modifie(elements, noeuds)
    # Tout est rationnel en les composants et en `s` : l'elimination se fait
    # dans le corps des fractions rationnelles, exactement, la ou
    # `simplify` s'enlise des la cinquieme inconnue.
    augmentee = DomainMatrix.from_Matrix(matrice.row_join(membre)).to_field()
    reduite, pivots = augmentee.rref()
    if pivots != tuple(range(matrice.cols)):
        msg = "systeme nodal singulier : le circuit n'a pas de solution unique"
        raise ValueError(msg)
    return reduite.to_Matrix()[:, -1].applyfunc(sp.cancel)
