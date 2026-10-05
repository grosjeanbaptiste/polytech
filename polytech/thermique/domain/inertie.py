"""L'inertie thermique d'un solide isotherme.

Syllabus de thermique, tome 1, premiere partie, chapitre 3. Un solide
assez conducteur pour rester a temperature uniforme, plonge dans un
fluide, obeit a une equation ordinaire, `rho c V dt/dtau = K S (t_f - t)`.
Avec la resistance `R = 1 / (K S)` et la capacite `C = rho c V`, c'est la
decharge d'un condensateur : l'ecart au fluide decroit en
`exp(-tau / tau_c)`, de constante de temps `tau_c = R C`.

Sa transmittance, `[t] / [t_f] = 1 / (1 + p R C)`, est celle d'un
systeme du premier ordre : la reponse a une rampe, l'erreur de trainage
`B R C`, l'attenuation et le dephasage d'une sollicitation periodique
sont ceux que le cours de signaux etablit, et ils en viennent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import sympy as sp

from polytech.signaux.domain.premier_ordre import PremierOrdre

type Grandeur = sp.Expr | float


@dataclass(frozen=True, slots=True)
class SolideIsotherme:
    """La resistance de surface `R` et la capacite thermique `C`."""

    resistance: Grandeur
    capacite: Grandeur

    @classmethod
    def depuis(
        cls, *, echange: float, surface: float, capacite: float
    ) -> SolideIsotherme:
        """`R = 1 / (K S)`, la capacite `rho c V` etant donnee."""
        if surface <= 0:
            msg = f"surface d'echange nulle ou negative : {surface} m^2"
            raise ValueError(msg)
        return cls(resistance=1 / (echange * surface), capacite=capacite)

    def constante_de_temps(self) -> Grandeur:
        """`tau_c = R C`."""
        return self.resistance * self.capacite

    def premier_ordre(self) -> PremierOrdre:
        """Le systeme du premier ordre `1 / (1 + p R C)`."""
        return PremierOrdre(gain=sp.S.One, constante=self.constante_de_temps())

    def ecart_reduit(self, duree: Grandeur) -> sp.Expr:
        """`theta / theta_0 = exp(-tau / tau_c)`."""
        return sp.exp(-duree / self.constante_de_temps())

    def duree_pour(self, rapport: float) -> float:
        """Le temps au bout duquel l'ecart est reduit a ce rapport."""
        if not 0 < rapport <= 1:
            msg = f"rapport hors de ]0, 1] : {rapport}"
            raise ValueError(msg)
        return -float(self.constante_de_temps()) * math.log(rapport)

    def erreur_de_trainage(self, pente: Grandeur) -> Grandeur:
        """`B R C` : le retard permanent sur un fluide qui monte en rampe."""
        return pente * self.constante_de_temps()

    def reponse_periodique(self, pulsation: float) -> tuple[float, float]:
        """L'attenuation et le dephasage, en radians, a cette pulsation."""
        gain, phase = self.premier_ordre().regime_harmonique(sp.Float(pulsation))
        return (float(gain), float(phase))
