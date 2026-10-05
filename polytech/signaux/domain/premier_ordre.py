"""Le systeme du premier ordre et ses reponses.

Syllabus de signaux et systemes, section 5.1. Sa transmittance n'a qu'un
pole : `H(p) = K / (p T + 1)` en forme de Bode, `K_e / (p + a)` en forme
d'Evans, avec `K_e = K / T` et `a = 1 / T`. C'est le cas d'un circuit
passif qui ne porte qu'un element reactif.

Toutes ses reponses font paraitre la meme exponentielle, de constante de
temps `T` : `h = (K / T) exp(-t / T)`, `a = K (1 - exp(-t / T))`, et
sous une rampe `K t - K T (1 - exp(-t / T))`, en retard de `T` pour de
bon -- l'erreur de trainage `K T`.

Sous une sinusoide, la reponse forcee a deux termes. Celui de regime ne
differe de l'entree que par un gain et un dephasage ; le transitoire,
signature du pole, n'est la que pour rendre la sortie compatible avec
l'etat initial.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.symboles import TEMPS as T
from polytech.signaux.domain.transmittance import Transmittance


@dataclass(frozen=True, slots=True)
class PremierOrdre:
    """Le gain statique `K` et la constante de temps `T`."""

    gain: sp.Expr
    constante: sp.Expr

    def __post_init__(self) -> None:
        """Refuse une constante de temps qui ne serait pas un temps."""
        if sp.sympify(self.constante).is_nonpositive:
            msg = f"constante de temps nulle ou negative : {self.constante}"
            raise ValueError(msg)

    def de_bode(self) -> Transmittance:
        """`K / (p T + 1)`."""
        return Transmittance(expression=self.gain / (P * self.constante + 1))

    def d_evans(self) -> tuple[sp.Expr, sp.Expr]:
        """`(K_e, a)` de `K_e / (p + a)`."""
        inverse = 1 / sp.sympify(self.constante)
        return (self.gain * inverse, inverse)

    def impulsionnelle(self) -> sp.Expr:
        """`h(t) = (K / T) exp(-t / T)`."""
        return self.gain / self.constante * sp.exp(-T / self.constante)

    def indicielle(self) -> sp.Expr:
        """`a(t) = K (1 - exp(-t / T))`."""
        return self.gain * (1 - sp.exp(-T / self.constante))

    def reponse_a_la_rampe(self) -> sp.Expr:
        """`y(t) = K t - K T (1 - exp(-t / T))`."""
        return self.gain * T - self.erreur_de_trainage() * self.indicielle() / self.gain

    def erreur_de_trainage(self) -> sp.Expr:
        """`K T` : le retard permanent sur la rampe, aussi dite de poursuite."""
        return self.gain * self.constante

    def regime_harmonique(self, pulsation: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
        """Le gain `K / racine(omega^2 T^2 + 1)` et le dephasage `-arctan(omega T)`."""
        reduite = pulsation * self.constante
        return (self.gain / sp.sqrt(reduite**2 + 1), -sp.atan(reduite))

    def reponse_au_cosinus(self, pulsation: sp.Expr, phase: sp.Expr = 0) -> sp.Expr:
        """La reponse a `cos(omega t + phase)`, appliquee en `t = 0`."""
        gain, dephasage = self.regime_harmonique(pulsation)
        regime = sp.cos(pulsation * T + phase + dephasage)
        transitoire = sp.exp(-T / self.constante) * sp.cos(phase + dephasage)
        return gain * (regime - transitoire)
