"""Les performances d'une boucle fermee, telles que le cours les mesure.

Ancrage direct sur `Exs_08_Performances_Reponses` : la feuille publie
`|M(0)| = -3 dB` et `y(infini) = 0,707`, puis `D = 87 %` et
`y_max = 1,32`. Les deux couples se recoupent exactement -- `10^(-3/20)`
vaut `0,7079`, et `0,707 x 1,87` vaut `1,322` -- et ce sont ces deux
relations que ce module porte.

Ce qu'il ne porte **pas**, et c'est delibere : le passage du pic de
resonance au depassement. Le cours le lit sur les abaques de
Black-Nichols, qui sont un abaque et non une formule ; le seul lien en
forme close est celui du second ordre, et il est expose comme tel. Deduire
`D` d'un pic par la formule du second ordre sur un systeme d'ordre trois
serait une approximation non declaree.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from polytech.signaux.domain.harmonique import depuis_les_decibels

if TYPE_CHECKING:
    import sympy as sp


def gain_statique_depuis_decibels(decibels: sp.Expr) -> sp.Expr:
    """`|M(0)|` en dB donne la valeur finale de la reponse indicielle."""
    return depuis_les_decibels(decibels)


def erreur_statique_de_consigne(gain_de_boucle: float) -> float:
    """`1 / (1 + K)` : ce qui reste de l'ecart en regime permanent.

    Un gain infini -- c'est-a-dire un integrateur dans la boucle --
    l'annule exactement. C'est le resultat que le fascicule met en avant,
    et c'est la raison d'etre du terme integral d'un correcteur.
    """
    if gain_de_boucle < 0:
        msg = f"gain de boucle negatif : {gain_de_boucle}"
        raise ValueError(msg)
    if math.isinf(gain_de_boucle):
        return 0.0
    return 1 / (1 + gain_de_boucle)


def erreur_de_perturbation(
    amplitude: float,
    gain_aval: float,
    gain_de_boucle: float,
) -> float:
    """L'effet d'une perturbation, attenue par la boucle mais pas annule.

    Le fascicule insiste : une perturbation qui entre **en aval** du
    correcteur n'est pas rejetee comme une erreur de consigne. Son effet
    vaut `A G_aval / (1 + K)`, et le gain aval ne se laisse pas oublier.
    """
    return amplitude * gain_aval * erreur_statique_de_consigne(gain_de_boucle)


def module_en_continu(gain_de_boucle: float) -> float:
    """`K / (1 + K)` : le gain de la boucle fermee en continu, note `|M(0)|`."""
    if gain_de_boucle < 0:
        msg = f"gain de boucle negatif : {gain_de_boucle}"
        raise ValueError(msg)
    return gain_de_boucle / (1 + gain_de_boucle)
