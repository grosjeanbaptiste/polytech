"""La boucle fermee et ses transmittances equivalentes.

Planches « Boucle fermee et stabilite » du cours de signaux et systemes.
Mesurer la sortie, la comparer a la consigne, agir selon l'ecart : la
chaine directe porte le controleur `C`, l'actionneur `G_1` et le procede
`G_2`, sur lequel la perturbation agit ; la chaine de reaction porte le
capteur `G_s`. La boucle se resume a deux transmittances,

`T_R = C G_1 G_2 / (1 + G_s C G_1 G_2)` vis-a-vis de la consigne,
`T_P = -G_2 / (1 + G_s C G_1 G_2)` vis-a-vis de la perturbation,

et `Y = T_R R + T_P P`. Elles ont le meme denominateur : la boucle est
stable si les racines de l'equation caracteristique
`1 + G_s C G_1 G_2 = 0` sont toutes a gauche de l'axe imaginaire.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp

from polytech.signaux.domain.symboles import IMAGE as P


@dataclass(frozen=True, slots=True)
class Boucle:
    """Le controleur, l'actionneur, le procede et le capteur."""

    correcteur: sp.Expr
    actionneur: sp.Expr
    procede: sp.Expr
    capteur: sp.Expr

    def _directe(self) -> sp.Expr:
        return self.correcteur * self.actionneur * self.procede

    def ouverte(self) -> sp.Expr:
        """`G_s C G_1 G_2`, la fonction de transfert en boucle ouverte."""
        return self.capteur * self._directe()

    def vis_a_vis_de_la_consigne(self) -> sp.Expr:
        """`T_R = C G_1 G_2 / (1 + G_s C G_1 G_2)`."""
        return sp.cancel(self._directe() / (1 + self.ouverte()))

    def vis_a_vis_de_la_perturbation(self) -> sp.Expr:
        """`T_P = -G_2 / (1 + G_s C G_1 G_2)`."""
        return sp.cancel(-self.procede / (1 + self.ouverte()))

    def sortie(self, consigne: sp.Expr, perturbation: sp.Expr) -> sp.Expr:
        """`Y = T_R R + T_P P`."""
        par_la_consigne = self.vis_a_vis_de_la_consigne() * consigne
        return par_la_consigne + self.vis_a_vis_de_la_perturbation() * perturbation

    def est_stable(self) -> bool:
        """Vrai si toutes les racines de `1 + G_s C G_1 G_2` sont a gauche."""
        caracteristique = sp.fraction(sp.cancel(1 + self.ouverte()))[0]
        coefficients = [complex(c) for c in sp.Poly(caracteristique, P).all_coeffs()]
        return bool(np.all(np.roots(coefficients).real < 0))
