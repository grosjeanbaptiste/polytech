"""L'analyse harmonique : les trois lectures d'une meme reponse frequentielle.

Bode separe module et argument sur deux diagrammes ; Nyquist les trace
dans le plan complexe ; Black porte le module en ordonnee et l'argument en
abscisse. Ce sont trois **lectures** du meme `G(j omega)`, et le cours
insiste : aucune n'apporte d'information que les deux autres n'aient.

Les marges sont definies la ou le cours les definit : la marge de gain au
croisement de l'axe reel negatif, la marge de phase au croisement du
cercle unite. Les confondre est l'erreur classique, et le corrige officiel
du cours en porte d'ailleurs la trace -- il ecrit « pour une marge de gain
de 60 degres », alors qu'une marge de gain ne s'exprime pas en degres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.signaux.domain.symboles import PULSATION

if TYPE_CHECKING:
    from polytech.signaux.domain.transmittance import Transmittance

DECIBELS_PAR_DECADE = 20
DEGRES_PAR_RADIAN = 180


def en_decibels(module: sp.Expr) -> sp.Expr:
    """`20 log10(|G|)`, la mesure du cours."""
    return DECIBELS_PAR_DECADE * sp.log(module, 10)


def depuis_les_decibels(decibels: sp.Expr) -> sp.Expr:
    """L'inverse : du gain en decibels au module."""
    return sp.Integer(10) ** (sp.Rational(decibels) / DECIBELS_PAR_DECADE)


def en_degres(radians: sp.Expr) -> sp.Expr:
    """L'argument en degres, comme le cours le lit sur ses abaques."""
    return radians * DEGRES_PAR_RADIAN / sp.pi


@dataclass(frozen=True, slots=True)
class AnalyseHarmonique:
    """La reponse frequentielle d'une transmittance, et ses trois lectures."""

    transmittance: Transmittance

    def _en(self, pulsation: sp.Expr) -> sp.Expr:
        frequentielle = self.transmittance.reponse_frequentielle()
        return sp.simplify(frequentielle.subs(PULSATION, pulsation))

    def module(self, pulsation: sp.Expr) -> sp.Expr:
        """`|G(j omega)|` a cette pulsation."""
        return sp.simplify(sp.Abs(self._en(pulsation)))

    def argument(self, pulsation: sp.Expr) -> sp.Expr:
        """`arg G(j omega)` en radians."""
        return sp.simplify(sp.arg(self._en(pulsation)))

    def point_de_bode(self, pulsation: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
        """Le couple (gain en dB, argument en degres) que Bode porte."""
        return (
            sp.simplify(en_decibels(self.module(pulsation))),
            sp.simplify(en_degres(self.argument(pulsation))),
        )

    def point_de_nyquist(self, pulsation: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
        """Le couple (partie reelle, partie imaginaire) que Nyquist trace."""
        valeur = self._en(pulsation)
        return sp.simplify(sp.re(valeur)), sp.simplify(sp.im(valeur))

    def point_de_black(self, pulsation: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
        """Le couple (argument en degres, gain en dB) : Black inverse les axes."""
        gain, phase = self.point_de_bode(pulsation)
        return phase, gain
