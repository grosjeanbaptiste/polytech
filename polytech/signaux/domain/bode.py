"""Le trace des courbes frequentielles d'un systeme d'ordre quelconque.

Planches « Analyse harmonique » du cours de signaux et systemes. Mise
sous forme de Bode, une transmittance est un produit de facteurs
elementaires : un gain `K`, `alpha` integrateurs -- la classe du
systeme --, des poles et des zeros reels `p T + 1`, des paires
`p^2 T^2 + 2 xi p T + 1`. En decibels les modules s'ajoutent, et les
arguments aussi : les courbes de Bode sont la somme des courbes
elementaires, celles d'un zero etant l'opposee de celles d'un pole.

Le trace asymptotique en decoule : une pente de `-20 alpha` dB par
decade au depart, qui change de `-20` ou `+20` a chaque cassure simple,
du double a chaque paire.

Un second ordre resonne en `w_r = racine(1 - 2 xi^2) / T` si
`xi < racine(2) / 2`. Un delai ne touche pas au module et retire `w tau`
a l'argument, en radians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import PULSATION as W

if TYPE_CHECKING:
    from collections.abc import Sequence

type Courbe = tuple[sp.Expr, sp.Expr]
"""Le module et l'argument, en radians, en fonction de la pulsation."""

DECADE = 20
QUART_DE_TOUR = 90


def courbe_inverse(courbe: Courbe) -> Courbe:
    """Celle du facteur inverse : decibels et argument changent de signe."""
    return (1 / courbe[0], -courbe[1])


def courbe_du_delai(duree: sp.Expr) -> Courbe:
    """`exp(-j w tau)` : module un, argument `-w tau`."""
    return (sp.S.One, -W * duree)


def somme_des_courbes(courbes: Sequence[Courbe]) -> Courbe:
    """La courbe du produit : les modules se multiplient, les arguments s'ajoutent."""
    if not courbes:
        msg = "aucune courbe elementaire : il n'y a rien a sommer"
        raise ValueError(msg)
    module = sp.Mul(*(courbe[0] for courbe in courbes))
    return (module, sum((courbe[1] for courbe in courbes), sp.S.Zero))


def classe_du_systeme(transmittance: sp.Expr) -> int:
    """`alpha`, le nombre d'integrateurs : la multiplicite du pole en zero."""
    denominateur = sp.fraction(sp.cancel(transmittance))[1]
    return int(sp.roots(sp.Poly(denominateur, P)).get(sp.S.Zero, 0))


def depart_et_arrivee_de_nyquist(transmittance: sp.Expr) -> tuple[int, int]:
    """Les arguments, en degres, aux basses puis aux hautes frequences."""
    numerateur, denominateur = (
        sp.Poly(e, P) for e in sp.fraction(sp.cancel(transmittance))
    )
    exces = denominateur.degree() - numerateur.degree()
    if exces <= 0:
        msg = "degre du numerateur non inferieur : la courbe n'arrive pas a l'origine"
        raise ValueError(msg)
    depart = -QUART_DE_TOUR * classe_du_systeme(transmittance)
    return (depart, -QUART_DE_TOUR * exces)


def pulsation_de_resonance(constante: float, amortissement: float) -> float:
    """`w_r = racine(1 - 2 xi^2) / T`."""
    if 2 * amortissement**2 >= 1:
        msg = f"pas de resonance : xi = {amortissement} atteint racine(2) / 2"
        raise ValueError(msg)
    return math.sqrt(1 - 2 * amortissement**2) / constante


def pulsation_de_coupure_du_second_ordre(
    constante: float, amortissement: float
) -> float:
    """La pulsation ou le module tombe a `K / racine(2)`."""
    carre = amortissement**2
    interieur = 1 - 2 * carre + math.sqrt(4 * carre**2 - 4 * carre + 2)
    return math.sqrt(interieur) / constante


@dataclass(frozen=True, slots=True)
class TraceAsymptotique:
    """Le gain, la classe, et les cassures : (pulsation, pente en multiples de 20)."""

    gain: float
    classe: int
    cassures: tuple[tuple[float, int], ...]

    def pentes(self) -> tuple[int, ...]:
        """Les pentes successives, en dB par decade."""
        pentes = [-DECADE * self.classe]
        for _, saut in sorted(self.cassures):
            pentes.append(pentes[-1] + DECADE * saut)
        return tuple(pentes)

    def gain_en_decibels(self, pulsation: float) -> float:
        """L'ordonnee de l'asymptote a cette pulsation."""
        if pulsation <= 0:
            msg = f"pulsation nulle ou negative : {pulsation} rad/s"
            raise ValueError(msg)
        depart = math.log10(self.gain) - self.classe * math.log10(pulsation)
        decibels = DECADE * depart
        for cassure, saut in self.cassures:
            if pulsation > cassure:
                decibels += DECADE * saut * math.log10(pulsation / cassure)
        return decibels
