"""Les signaux elementaires, et la classe a laquelle chacun appartient.

Le cours ouvre sur cette classification, et elle n'est pas decorative :
elle decide de ce que l'on peut calculer. Un signal d'**energie finie** a
une energie et une puissance moyenne nulle ; un signal de **puissance
finie** a une puissance et une energie infinie. Confondre les deux fait
calculer une energie qui diverge sans que rien le signale -- sympy rend
`oo`, qui se compose parfaitement.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import sympy as sp

from polytech.signaux.domain.symboles import TEMPS


class Classe(Enum):
    """Les deux classes du cours, et celle qui n'en est aucune."""

    ENERGIE_FINIE = "energie finie"
    PUISSANCE_FINIE = "puissance finie"
    NI_L_UN_NI_L_AUTRE = "ni energie ni puissance finie"


def echelon() -> sp.Expr:
    """L'echelon unite, defini sur `t >= 0` comme le cours le pose."""
    return sp.Heaviside(TEMPS)


def rampe() -> sp.Expr:
    """La rampe unite `t u(t)`."""
    return TEMPS * sp.Heaviside(TEMPS)


@dataclass(frozen=True, slots=True)
class Signal:
    """Un signal causal, donne par son expression en `t`."""

    expression: sp.Expr

    def sur_le_domaine(self) -> sp.Expr:
        """L'expression reduite sur `t >= 0`, ou l'echelon unite vaut un.

        Les deux integrales de ce module courent de zero : le facteur
        `Heaviside(t)` y vaut donc un partout, et le garder empeche
        seulement sympy de conclure -- il ne sait pas integrer
        `Heaviside(t)^2`. C'est la meme decision que pour l'inversion de
        Laplace dans `polytech.maths`, et pour la meme raison : le signal
        est causal, le domaine est la demi-droite positive.

        Un echelon **retarde** `Heaviside(t - tau)` n'est pas reduit : il
        dit a quel instant le signal commence, ce qui est une information
        de l'enonce et non une convention de domaine.
        """
        return self.expression.subs(sp.Heaviside(TEMPS), 1)

    def energie(self) -> sp.Expr:
        """L'integrale de `|x|^2` sur `[0, oo)` ; peut diverger."""
        return sp.integrate(sp.Abs(self.sur_le_domaine()) ** 2, (TEMPS, 0, sp.oo))

    def puissance_moyenne(self) -> sp.Expr:
        """La limite de l'energie sur la duree, quand la duree croit."""
        duree = sp.Symbol("T", positive=True)
        reduite = sp.Abs(self.sur_le_domaine()) ** 2
        partielle = sp.integrate(reduite, (TEMPS, 0, duree))
        return sp.limit(partielle / duree, duree, sp.oo)

    def classe(self) -> Classe:
        """La classe du cours a laquelle ce signal appartient."""
        if self.energie().is_finite:
            return Classe.ENERGIE_FINIE
        if self.puissance_moyenne().is_finite:
            return Classe.PUISSANCE_FINIE
        return Classe.NI_L_UN_NI_L_AUTRE

    def energie_verifiee(self) -> sp.Expr:
        """L'energie ; leve si elle diverge, au lieu de rendre `oo`."""
        energie = self.energie()
        if not energie.is_finite:
            msg = (
                f"energie infinie : {self.expression} n'est pas un signal "
                f"d'energie finie (classe {self.classe().value})"
            )
            raise ValueError(msg)
        return energie
