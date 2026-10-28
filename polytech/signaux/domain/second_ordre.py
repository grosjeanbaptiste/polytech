"""Le systeme du second ordre : ses parametres et ses poles.

Syllabus de signaux et systemes, section 5.2. Deux ecritures de la meme
transmittance : `K / (p^2 T^2 + 2 xi p T + 1)` en forme de Bode,
`K_e / (p^2 + 2 sigma p + rho^2)` en forme d'Evans, avec `K_e = K / T^2`,
`sigma = xi / T` et `rho = 1 / T`.

Les poles disent tout. `-sigma` est leur moyenne arithmetique et `rho`
leur moyenne geometrique ; `xi` en est le rapport. Complexes conjugues
quand `xi < 1`, ils ont pour module `rho`, pour partie reelle `-sigma`,
pour partie imaginaire la pulsation propre `racine(1 - xi^2) / T`, et
`xi` est le sinus de leur argument compte depuis l'axe imaginaire. Reels
quand `xi >= 1`, ils valent `-1 / T_1` et `-1 / T_2`.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from polytech.signaux.domain.symboles import IMAGE as P
from polytech.signaux.domain.transmittance import Transmittance


@dataclass(frozen=True, slots=True)
class SecondOrdre:
    """Le gain statique `K`, la constante `T` et l'amortissement `xi`."""

    gain: sp.Expr
    constante: sp.Expr
    amortissement: sp.Expr

    def __post_init__(self) -> None:
        """Refuse un amortissement negatif : le systeme serait instable."""
        if sp.sympify(self.amortissement).is_negative:
            msg = f"amortissement negatif : {self.amortissement}"
            raise ValueError(msg)

    @classmethod
    def depuis_evans(cls, gain: sp.Expr, sigma: sp.Expr, rho: sp.Expr) -> SecondOrdre:
        """`K = K_e / rho^2`, `T = 1 / rho`, `xi = sigma / rho`."""
        return cls(gain / rho**2, 1 / rho, sigma / rho)

    @classmethod
    def depuis_les_constantes_de_temps(
        cls, gain: sp.Expr, premiere: sp.Expr, seconde: sp.Expr
    ) -> SecondOrdre:
        """`T = racine(T_1 T_2)`, `xi = (T_1 + T_2) / (2 racine(T_1 T_2))`."""
        moyenne = sp.sqrt(premiere * seconde)
        return cls(gain, moyenne, (premiere + seconde) / (2 * moyenne))

    def de_bode(self) -> Transmittance:
        """`K / (p^2 T^2 + 2 xi p T + 1)`."""
        reduite = P * self.constante
        denominateur = reduite**2 + 2 * self.amortissement * reduite + 1
        return Transmittance(expression=self.gain / denominateur)

    def d_evans(self) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
        """`(K_e, sigma, rho)`."""
        inverse = 1 / sp.sympify(self.constante)
        return (self.gain * inverse**2, self.amortissement * inverse, inverse)

    def poles(self) -> tuple[sp.Expr, sp.Expr]:
        """`(-xi -/+ racine(xi^2 - 1)) / T`."""
        ecart = sp.sqrt(sp.sympify(self.amortissement) ** 2 - 1)
        return (
            (-self.amortissement - ecart) / self.constante,
            (-self.amortissement + ecart) / self.constante,
        )

    def regime(self) -> str:
        """La nature des poles, que la valeur de `xi` decide."""
        if self.amortissement < 1:
            return "poles complexes conjugues"
        # `sp.Float(1.0) == 1` est faux : on compare l'ecart a zero.
        confondus = sp.sympify(self.amortissement) - 1 == 0
        return "poles reels confondus" if confondus else "poles reels distincts"

    def pulsation_propre(self) -> sp.Expr:
        """`omega = racine(1 - xi^2) / T`, la partie imaginaire des poles."""
        if self.amortissement >= 1:
            msg = f"poles reels : pas de pulsation propre, xi = {self.amortissement}"
            raise ValueError(msg)
        return sp.sqrt(1 - sp.sympify(self.amortissement) ** 2) / self.constante

    def argument_des_poles(self) -> sp.Expr:
        """`theta`, tel que `sin(theta) = xi`."""
        return sp.asin(self.amortissement)

    def facteur_de_qualite(self) -> sp.Expr:
        """`Q = rho / (2 sigma) = 1 / (2 xi)`."""
        if self.amortissement == 0:
            msg = "facteur de qualite infini : l'amortissement est nul"
            raise ValueError(msg)
        return 1 / (2 * sp.sympify(self.amortissement))
