"""Enonce C -- efforts hydrostatiques sur un parement.

La poussee est obtenue par *integration* de la pression relative sur la
surface mouillee :

    F = int_0^H rho g z * largeur(z) dz

Chaque geometrie ne declare donc que sa largeur a la profondeur ``z`` ; la
formule, son exposant en ``H`` et la valeur numerique en decoulent tous, sans
qu'aucune expression ne soit recopiee.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import sympy as sp

from polytech.fluides.domain.grandeurs import PESANTEUR, Hauteur, MasseVolumique
from polytech.fluides.domain.statique.symboles import RHO, G, H, Z


def _exiger_fraction(fraction: sp.Expr | float) -> None:
    if not 0 < float(fraction) <= 1:
        msg = f"fraction de volume hors de ]0, 1] : {fraction}"
        raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class Parement(ABC):
    """Surface mouillee sur laquelle s'exerce la poussee."""

    fluide: MasseVolumique
    pesanteur: float = PESANTEUR

    @abstractmethod
    def largeur_mouillee(self) -> sp.Expr:
        """Largeur mouillee a la profondeur ``Z``, hauteur d'eau ``H``."""

    @abstractmethod
    def exposant_volume(self) -> int:
        """Exposant ``n`` tel que le volume stocke varie comme ``H^n``."""

    @abstractmethod
    def _valeurs(self) -> dict[sp.Symbol, float]:
        """Substitutions numeriques propres a la geometrie."""

    def expression_poussee(self) -> sp.Expr:
        """Resultante litterale, integree sur la surface mouillee."""
        return sp.simplify(
            sp.integrate(RHO * G * Z * self.largeur_mouillee(), (Z, 0, H))
        )

    def expression_moment(self) -> sp.Expr:
        """Moment de renversement litteral autour du pied."""
        return sp.simplify(self.expression_poussee() * H / 3)

    def exposant_poussee(self) -> int:
        """Degre en ``H`` de la poussee, lu sur l'integrale elle-meme."""
        return int(sp.degree(sp.expand(self.expression_poussee()), H))

    def centre_de_poussee(self, hauteur: Hauteur) -> float:
        """Altitude du centre de poussee au-dessus du fond, en metres."""
        return hauteur.metres / 3.0

    def _evaluer(self, expression: sp.Expr, hauteur: Hauteur) -> float:
        valeurs = {
            RHO: self.fluide.kg_par_m3,
            G: self.pesanteur,
            H: hauteur.metres,
            **self._valeurs(),
        }
        return float(expression.subs(valeurs))

    def poussee(self, hauteur: Hauteur) -> float:
        """Resultante numerique, evaluee sur l'integrale litterale."""
        return self._evaluer(self.expression_poussee(), hauteur)

    def moment(self, hauteur: Hauteur) -> float:
        """Moment numerique, evalue sur l'expression litterale."""
        return self._evaluer(self.expression_moment(), hauteur)

    def hauteur_pour_fraction_de_volume(self, fraction: sp.Expr | float) -> sp.Expr:
        """Rapport ``H'/H`` quand le volume est ramene a ``fraction``."""
        _exiger_fraction(fraction)
        return sp.nsimplify(fraction) ** sp.Rational(1, self.exposant_volume())

    def rapport_poussee(self, fraction: sp.Expr | float) -> sp.Expr:
        """Rapport ``F'/F`` exact, sous forme rationnelle."""
        return sp.nsimplify(
            self.hauteur_pour_fraction_de_volume(fraction) ** self.exposant_poussee(),
        )

    def rapport_moment(self, fraction: sp.Expr | float) -> sp.Expr:
        """Rapport ``M'/M`` exact : un degre de plus que la poussee."""
        return sp.nsimplify(
            self.hauteur_pour_fraction_de_volume(fraction)
            ** (self.exposant_poussee() + 1),
        )
