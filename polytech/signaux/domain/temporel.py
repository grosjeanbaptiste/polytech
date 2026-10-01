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


def reponse_a_une_rampe(transmittance: Transmittance) -> sp.Expr:
    """L'original de `T(p) / p^2` : la reponse a une rampe unite.

    Le resultat que le fascicule met en avant : un premier ordre suit une
    rampe avec un retard permanent **exactement** egal a sa constante de
    temps. C'est ce qui rend l'erreur de trainage calculable sans
    simulation.
    """
    inversee = sp.inverse_laplace_transform(
        transmittance.expression / IMAGE**2,
        IMAGE,
        TEMPS,
        noconds=True,
    )
    return sp.simplify(inversee.subs(sp.Heaviside(TEMPS), 1))


def reponse_harmonique(transmittance: Transmittance) -> sp.Expr:
    """`T(j omega)` : le regime permanent sinusoidal."""
    return transmittance.reponse_frequentielle()


def ordre_du_systeme(transmittance: Transmittance) -> int:
    """Le degre du denominateur : la classification du fascicule.

    Il structure tout le chapitre dessus -- premier ordre, second ordre,
    troisieme ordre, ordre quelconque -- parce que la forme de la reponse
    en depend qualitativement et non seulement quantitativement.
    """
    return transmittance.degre()


ORDRE_TROIS = 3
"""Le fascicule consacre une partie entiere au troisieme ordre."""


def est_du_troisieme_ordre(transmittance: Transmittance) -> bool:
    """Vrai si le denominateur est de degre trois.

    Le fascicule lui consacre une partie parce que trois poles donnent
    trois modes, dont au plus un reel isole et une paire conjuguee -- et
    que l'allure resultante ne se devine pas depuis le second ordre.
    """
    return ordre_du_systeme(transmittance) == ORDRE_TROIS
