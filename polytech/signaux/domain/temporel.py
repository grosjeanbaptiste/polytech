"""Les reponses temporelles, et le theoreme de la valeur finale.

Le cours distingue la reponse **impulsionnelle** -- l'original de la
transmittance elle-meme -- et la reponse **indicielle**, qui est celle de
`T(p)/p`. Les deux se tirent de la meme transmittance, et c'est ce que le
corrige doit montrer plutot que de les calculer separement.

La valeur finale se lit sans inverser quoi que ce soit : `lim p T(p)/p`
quand `p` tend vers zero, soit `T(0)`. Le theoreme n'est valable que si le
regime final existe -- donc si la boucle est stable -- et la garde le dit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE, TEMPS

if TYPE_CHECKING:
    from polytech.signaux.domain.transmittance import Transmittance


@dataclass(frozen=True, slots=True)
class ReponsesTemporelles:
    """Les reponses d'une transmittance aux entrees canoniques du cours."""

    transmittance: Transmittance

    def impulsionnelle(self) -> sp.Expr:
        """L'original de `T(p)` : la reponse a une impulsion de Dirac."""
        inversee = sp.inverse_laplace_transform(
            self.transmittance.expression,
            IMAGE,
            TEMPS,
            noconds=True,
        )
        return sp.simplify(inversee.subs(sp.Heaviside(TEMPS), 1))

    def indicielle(self) -> sp.Expr:
        """L'original de `T(p)/p` : la reponse a un echelon unite."""
        inversee = sp.inverse_laplace_transform(
            self.transmittance.expression / IMAGE,
            IMAGE,
            TEMPS,
            noconds=True,
        )
        return sp.simplify(inversee.subs(sp.Heaviside(TEMPS), 1))

    def valeur_finale(self) -> sp.Expr:
        """`T(0)`, la valeur finale de la reponse indicielle.

        Leve si un pole est a partie reelle positive ou nulle : le regime
        final n'existe alors pas, et `T(0)` serait un nombre sans
        signification que le corrige afficherait comme un resultat.
        """
        instables = [
            pole
            for pole in self.transmittance.poles()
            if pole.is_number and sp.re(pole) >= 0
        ]
        if instables:
            msg = (
                f"pas de regime final : poles a partie reelle positive ou "
                f"nulle {instables}"
            )
            raise ValueError(msg)
        return sp.simplify(self.transmittance.expression.subs(IMAGE, 0))
