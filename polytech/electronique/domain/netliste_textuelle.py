"""Un circuit ecrit en netliste, et le graphe oriente qu'il est.

Planches du chapitre 1, partie II (planches 27 et 28). Un circuit
electrique est d'abord un objet mathematique : un graphe oriente `G =
(V, E)`, dont les noeuds sont les interconnexions et dont chaque arete,
un couple ordonne de noeuds distincts, porte la loi d'un composant.

La **netliste** en est l'ecriture pour une machine, une ligne par
composant :

    r1 1 2 1e3

une lettre pour la nature, un numero qui l'identifie, le noeud de
depart, le noeud d'arrivee, et la valeur en unites du Systeme
international. Pour une source de tension, la borne positive est le
noeud d'arrivee.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.electronique.domain.netliste import (
    Admittance,
    Element,
    SourceDeTension,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

NATURES = ("r", "c", "l", "v")
"""Resistance, condensateur, inductance, source de tension."""
CHAMPS = 4


@dataclass(frozen=True, slots=True)
class Composant:
    """Une ligne de netliste : nature, numero, depart, arrivee, valeur."""

    nature: str
    numero: int
    depart: int
    arrivee: int
    valeur: float | sp.Expr


def _lire_la_ligne(ligne: str) -> Composant:
    champs = ligne.split()
    if len(champs) != CHAMPS:
        msg = f"ligne de netliste illisible : {ligne!r}"
        raise ValueError(msg)
    nom, depart, arrivee, valeur = champs
    if nom[0] not in NATURES:
        msg = f"composant inconnu : {nom!r}"
        raise ValueError(msg)
    return Composant(nom[0], int(nom[1:]), int(depart), int(arrivee), float(valeur))


def lire_la_netliste(texte: str) -> tuple[Composant, ...]:
    """Les composants, dans l'ordre des lignes ; les lignes vides sont sautees."""
    composants = tuple(
        _lire_la_ligne(ligne) for ligne in texte.splitlines() if ligne.strip()
    )
    noms = [(c.nature, c.numero) for c in composants]
    for nom in noms:
        if noms.count(nom) > 1:
            msg = f"identifiant deja employe : {nom[0]}{nom[1]}"
            raise ValueError(msg)
    return composants


def graphe_du_circuit(
    composants: Sequence[Composant],
) -> tuple[frozenset[int], tuple[tuple[int, int], ...]]:
    """`(V, E)` : les noeuds, et une arete orientee par composant."""
    for composant in composants:
        if composant.depart == composant.arrivee:
            msg = f"{composant.nature}{composant.numero} relie un noeud a lui-meme"
            raise ValueError(msg)
    aretes = tuple((c.depart, c.arrivee) for c in composants)
    return frozenset(noeud for arete in aretes for noeud in arete), aretes


def _admittance(composant: Composant, laplace: object) -> sp.Expr:
    valeur, variable = sp.sympify(composant.valeur), sp.sympify(laplace)
    if composant.nature == "r":
        return 1 / valeur
    if composant.nature == "c":
        return variable * valeur
    if variable == 0:
        msg = f"l{composant.numero} : une inductance est un court-circuit en continu"
        raise ValueError(msg)
    return 1 / (variable * valeur)


def elements_nodaux(
    composants: Sequence[Composant], *, laplace: object
) -> tuple[Element, ...]:
    """Les elements de l'analyse nodale, a la variable de Laplace donnee."""
    elements: list[Element] = []
    for composant in composants:
        if composant.nature == "v":
            courant = sp.Symbol(f"i_v{composant.numero}")
            valeur = sp.sympify(composant.valeur)
            elements.append(
                SourceDeTension(composant.arrivee, composant.depart, valeur, courant)
            )
            continue
        admittance = _admittance(composant, laplace)
        elements.append(Admittance(composant.depart, composant.arrivee, admittance))
    return tuple(elements)
