"""Annexe du TP 7, sections 1 a 3 : echelles LC et transformations.

L'annexe est un formulaire. Ce module en rejoue chaque regle sur un meme
filtre -- le prototype de Butterworth d'ordre 3, entre deux resistances
d'un ohm -- et recalcule ses deux tables.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

import sympy as sp

from polytech.electronique.domain import chebyshev, echelle_lc
from polytech.electronique.domain.abcd import (
    cascade,
    derivation,
    gain_en_tension,
    serie,
)
from polytech.electronique.domain.conversions_d_elements import (
    Branche,
    changer_la_coupure,
    en_passe_bande,
    en_passe_haut,
)
from polytech.electronique.domain.transpositions import mise_a_l_echelle_d_impedance

if TYPE_CHECKING:
    from collections.abc import Iterable

S: Final = sp.Symbol("s")
ORDRE: Final = 3
ONDULATION: Final = 0.1
"""L'ondulation de la table 2, en decibels."""
elements_de_butterworth = echelle_lc.elements_de_butterworth


def elements_de_chebyshev(ordre: int) -> tuple[float, ...]:
    """`g_1` a `g_N`, puis la charge `g_(N+1)`, comme la table 2 les donne."""
    elements = chebyshev.elements_de_chebyshev(ONDULATION, ordre)
    return (*elements, chebyshev.charge_de_chebyshev(ONDULATION, ordre))


def premier_element_exact() -> float:
    """`g_1` de Chebyshev a l'ordre 1, sans table : `2 epsilon`.

    `epsilon = racine(10^(L/10) - 1)` est l'ondulation ; a l'ordre 1 le
    filtre n'a qu'un element, qui la fixe a lui seul.
    """
    return 2 * math.sqrt(10 ** (ONDULATION / 10) - 1)


def _branches() -> tuple[Branche, ...]:
    """Le prototype : capacite en derivation, inductance en serie, capacite."""
    g1, g2, g3 = (round(g, 12) for g in elements_de_butterworth(ORDRE))
    return Branche(capacite=g1), Branche(inductance=g2), Branche(capacite=g3)


def _transfert(branches: Iterable[Branche], terminaison: float = 1.0) -> sp.Expr:
    """`V_charge / V_source` de l'echelle, entre deux resistances egales."""
    resistance = sp.nsimplify(terminaison, rational=True)
    etages = [serie(resistance)]
    for rang, branche in enumerate(branches):
        impedance = branche.impedance(S)
        etages.append(serie(impedance) if rang % 2 else derivation(impedance))
    return sp.cancel(sp.together(gain_en_tension(cascade(*etages), resistance)))


def prototype() -> sp.Expr:
    """Le transfert du prototype : coupure a 1 rad/s, 1/2 en continu."""
    return _transfert(_branches())


def exemple_de_mise_a_l_echelle() -> tuple[float, float, float]:
    """L'exemple de la page 1 : facteur 10, sur 5 microhenrys et 20 nanofarads."""
    return mise_a_l_echelle_d_impedance(10, 1.0, 5e-6, 20e-9)


def echelle_mise_a_l_echelle(facteur: float) -> sp.Expr:
    """Le prototype dont toutes les impedances sont multipliees par `facteur`."""
    branches = []
    for branche in _branches():
        _, inductance, capacite = mise_a_l_echelle_d_impedance(
            facteur, 1.0, branche.inductance or 1.0, branche.capacite or 1.0
        )
        if branche.inductance is None:
            branches.append(Branche(capacite=capacite))
        else:
            branches.append(Branche(inductance=inductance))
    return _transfert(branches, terminaison=facteur)


def vers_passe_bas(coupure: float) -> sp.Expr:
    """Le prototype, apres la transformation passe-bas vers passe-bas."""
    return _transfert(changer_la_coupure(b, coupure) for b in _branches())


def vers_passe_haut(coupure: float) -> sp.Expr:
    """Le prototype, apres la transformation passe-bas vers passe-haut."""
    return _transfert(en_passe_haut(b, coupure) for b in _branches())


def vers_passe_bande(centre: float, largeur: float) -> sp.Expr:
    """Le prototype, apres la transformation passe-bas vers passe-bande."""
    return _transfert(en_passe_bande(b, centre, largeur) for b in _branches())


def modules_du_prototype() -> tuple[float, float]:
    """`|T|` en continu, puis a la pulsation de coupure normalisee."""
    transfert = prototype()
    return float(sp.Abs(transfert.subs(S, 0))), float(sp.Abs(transfert.subs(S, sp.I)))
