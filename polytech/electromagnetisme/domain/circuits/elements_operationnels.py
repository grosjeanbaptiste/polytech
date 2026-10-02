"""Les elements d'un reseau en regime transitoire.

Separes du resolveur quand celui-ci a franchi la centaine de lignes. La
coupure suit une vraie frontiere : ici, ce que **sont** une bobine, une
capacite et un couplage ; la-bas, la facon d'en tirer des equations.

Les deux elements a memoire portent leur condition initiale. Elle vaut
zero par defaut, ce qui est le cas d'un circuit au repos avant la
commutation -- et ce qu'il faut verifier avant de s'en contenter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.electromagnetisme.domain.circuits.reseau import (
    Resistance,
    SourceDeCourant,
    SourceDeTension,
)

NUL: Final = sp.Integer(0)
"""Le zero exact, partage par les deux conditions initiales par defaut."""


@dataclass(frozen=True, slots=True)
class Bobine:
    """Une inductance, et le courant qui la parcourt a l'instant zero."""

    de: str
    vers: str
    inductance: sp.Expr
    courant_initial: sp.Expr = NUL


@dataclass(frozen=True, slots=True)
class Capacite:
    """Un condensateur, et sa tension `V(de) - V(vers)` a l'instant zero."""

    de: str
    vers: str
    capacite: sp.Expr
    tension_initiale: sp.Expr = NUL


@dataclass(frozen=True, slots=True)
class Couplage:
    """Une inductance mutuelle entre deux bobines du reseau."""

    premiere: Bobine
    seconde: Bobine
    mutuelle: sp.Expr


Element = Resistance | SourceDeTension | SourceDeCourant | Bobine | Capacite


def bornes(element: Element) -> tuple[str, str]:
    """Les deux noeuds de l'element, le premier etant sa borne d'entree."""
    if isinstance(element, SourceDeTension):
        return element.plus, element.moins
    return element.de, element.vers
