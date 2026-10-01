"""La stabilite en boucle fermee, par les deux chemins du cours.

Routh-Hurwitz regarde le **denominateur** de la boucle fermee et n'a
besoin d'aucune frequence. Nyquist regarde la **reponse frequentielle** de
la boucle ouverte et n'a besoin d'aucun polynome. Les deux doivent rendre
la meme limite de gain : c'est le recoupement que la feuille de reponses
du cours permet de verifier, puisqu'elle publie les deux cotes.

La garde sur le coefficient nul n'est pas formelle. Le critere de Routh
suppose tous les coefficients presents ; sur `1/p^3` le tableau est
degenere, et en tirer une limite donnerait un nombre faux sans le
paraitre.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE, PULSATION

if TYPE_CHECKING:
    from collections.abc import Mapping

    from polytech.signaux.domain.transmittance import Transmittance


@dataclass(frozen=True, slots=True)
class Stabilite:
    """La stabilite de la boucle fermee d'une transmittance donnee."""

    boucle_ouverte: Transmittance

    def _coefficients(self) -> list[sp.Expr]:
        """Les coefficients du denominateur de la boucle fermee, degre decroissant."""
        denominateur = self.boucle_ouverte.en_boucle_fermee().denominateur()
        return list(sp.Poly(denominateur, IMAGE).all_coeffs())

    def condition_de_routh(self) -> sp.Expr:
        """Pour un troisieme degre : `a_1 a_2 - a_0 a_3`, positif si stable."""
        coefficients = self._coefficients()
        attendu = 4
        if len(coefficients) != attendu or any(c == 0 for c in coefficients):
            msg = (
                f"coefficient nul ou degre different de trois : {coefficients} "
                f"-- le tableau de Routh y serait degenere"
            )
            raise ValueError(msg)
        trois, deux, un, zero = coefficients
        return sp.expand(deux * un - trois * zero)

    def gain_critique(self, gain: sp.Symbol) -> sp.Expr:
        """Le gain qui annule la condition de Routh : la limite de stabilite."""
        racines = sp.solve(sp.Eq(self.condition_de_routh(), 0), gain)
        return sp.simplify(racines[0])

    def est_stable(self, valeurs: Mapping[sp.Symbol, sp.Expr]) -> bool:
        """Vrai si la condition de Routh est positive pour ces valeurs."""
        return bool(self.condition_de_routh().subs(dict(valeurs)).is_positive)

    def pulsation_du_croisement(self) -> sp.Expr:
        """La pulsation ou la courbe de Nyquist coupe l'axe reel."""
        imaginaire = sp.im(
            sp.expand(sp.simplify(self.boucle_ouverte.reponse_frequentielle())),
        )
        racines = sp.solve(sp.Eq(sp.simplify(imaginaire), 0), PULSATION)
        positives = [r for r in racines if r.is_positive]
        if not positives:
            msg = "la courbe de Nyquist ne coupe pas l'axe reel"
            raise ValueError(msg)
        return sp.simplify(positives[0])

    def reel_au_croisement(self, valeurs: Mapping[sp.Symbol, sp.Expr]) -> sp.Expr:
        """L'abscisse du croisement, celle que le corrige lit sur la courbe."""
        frequentielle = self.boucle_ouverte.reponse_frequentielle()
        au_croisement = frequentielle.subs(PULSATION, self.pulsation_du_croisement())
        return sp.simplify(sp.re(sp.simplify(au_croisement.subs(dict(valeurs)))))

    def gain_critique_par_nyquist(self, gain: sp.Symbol) -> sp.Expr:
        """La limite tiree du module au croisement, sans aucun polynome.

        Nyquist dit que la boucle devient instable des que le point de
        croisement atteint `-1`. Si pour `K = 1` il vaut `-a`, la limite
        est `1 / a`.
        """
        unitaire = self.reel_au_croisement({gain: sp.Integer(1)})
        return sp.simplify(1 / abs(unitaire))
