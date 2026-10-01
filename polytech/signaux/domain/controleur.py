"""La synthese du controleur : de la specification aux parametres.

Le cours distingue explicitement l'**analyse** -- on connait `C G1 G2` et
on en deduit les performances -- de la **synthese**, ou `C` est inconnu et
seules les performances souhaitees sont donnees. Sa planche le dit :
« lors de la demarche de synthese du controleur seul `Gs G1 G2` est
connu ! ». C'est cette seconde direction que ce module porte, puisque la
premiere est deja celle de `harmonique` et de `performances`.

Un correcteur proportionnel-integral annule l'erreur statique parce qu'il
ajoute un pole a l'origine. La verification est donc intrinseque : apres
synthese, l'erreur statique doit valoir zero -- et le test l'exige au lieu
de le supposer.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE
from polytech.signaux.domain.transmittance import Transmittance


@dataclass(frozen=True, slots=True)
class Proportionnel:
    """`C(p) = K` : le plus simple, et celui qui laisse une erreur statique."""

    gain: sp.Expr

    def transmittance(self) -> Transmittance:
        """Le correcteur comme transmittance, pour l'assembler a la chaine."""
        return Transmittance(expression=self.gain)


@dataclass(frozen=True, slots=True)
class ProportionnelIntegral:
    """`C(p) = K (1 + 1/(T p))`, le correcteur PI du cours."""

    gain: sp.Expr
    temps_integral: sp.Expr

    def __post_init__(self) -> None:
        """Refuse un temps integral nul : il ferait exploser le correcteur."""
        if sp.simplify(self.temps_integral) == 0:
            msg = "temps integral nul : le correcteur PI n'est pas defini"
            raise ValueError(msg)

    def transmittance(self) -> Transmittance:
        """Le correcteur comme transmittance."""
        return Transmittance(
            expression=self.gain * (1 + 1 / (self.temps_integral * IMAGE)),
        )

    def erreur_statique(self, chaine: Transmittance) -> sp.Expr:
        """L'erreur statique de la boucle corrigee : `1 / (1 + C G)` en zero.

        Elle doit valoir zero -- c'est la raison d'etre de l'integrateur.
        La calculer plutot que l'affirmer est ce qui distingue une synthese
        verifiee d'une synthese esperee.
        """
        corrigee = self.transmittance().en_serie_avec(chaine)
        return sp.limit(1 / (1 + corrigee.expression), IMAGE, 0, "+")
