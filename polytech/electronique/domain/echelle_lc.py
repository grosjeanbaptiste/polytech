"""Le filtre en echelle LC : du tableau des `g_i` a la fonction de transfert.

Chapitre 3, partie I, planches 60 a 70. Un prototype passe-bas tout-poles
se realise par une echelle d'inductances en serie et de capacites en
derivation, entre une source de 1 ohm et une charge `g_(N+1)`. Deux
topologies existent, selon que l'echelle commence par une capacite en
derivation ou par une inductance en serie ; le meme tableau de `g_i` les
sert toutes deux, `g_i` etant une capacite ou une inductance selon sa
place.

Pour Butterworth, `g_i = 2 sin((2i - 1) pi / 2N)`, avec 3,01 dB de perte
d'insertion a 1 rad/s.
"""

from __future__ import annotations

import math
from enum import Enum

import sympy as sp

from polytech.electronique.domain.abcd import (
    cascade,
    derivation,
    gain_en_tension,
    serie,
)


class Topologie(Enum):
    """Le premier element apres la resistance de source."""

    CAPACITE_D_ABORD = "capacite en derivation"
    INDUCTANCE_D_ABORD = "inductance en serie"


def elements_de_butterworth(ordre: int) -> tuple[float, ...]:
    """`g_i = 2 sin((2i - 1) pi / 2N)`, pour `1 <= i <= N`."""
    if ordre < 1:
        msg = f"ordre inferieur a un : {ordre}"
        raise ValueError(msg)
    return tuple(
        2 * math.sin((2 * rang - 1) * math.pi / (2 * ordre))
        for rang in range(1, ordre + 1)
    )


def echelle(
    elements: tuple[float, ...],
    variable: sp.Expr,
    *,
    topologie: Topologie = Topologie.CAPACITE_D_ABORD,
) -> sp.Matrix:
    """La matrice de chaine de l'echelle, sans sa source ni sa charge."""
    if not elements:
        msg = "echelle vide : aucun element a mettre en cascade"
        raise ValueError(msg)
    capacite = topologie is Topologie.CAPACITE_D_ABORD
    etages = []
    for valeur in elements:
        exacte = sp.nsimplify(valeur, rational=True)
        if capacite:
            etages.append(derivation(1 / (exacte * variable)))
        else:
            etages.append(serie(exacte * variable))
        capacite = not capacite
    return cascade(*etages)


def transfert_de_l_echelle(
    elements: tuple[float, ...],
    variable: sp.Expr,
    *,
    topologie: Topologie = Topologie.CAPACITE_D_ABORD,
    source: float = 1.0,
    charge: float = 1.0,
) -> sp.Expr:
    """`V_charge / V_source`, source et charge comprises ; 1/2 en continu a 1 ohm."""
    if source < 0 or charge <= 0:
        msg = f"terminaisons invalides : source {source}, charge {charge} ohm"
        raise ValueError(msg)
    amont = serie(sp.nsimplify(source, rational=True))
    chaine = cascade(amont, echelle(elements, variable, topologie=topologie))
    gain = gain_en_tension(chaine, sp.nsimplify(charge, rational=True))
    return sp.cancel(sp.together(gain))
