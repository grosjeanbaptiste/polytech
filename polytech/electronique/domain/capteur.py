"""Le capteur : sa sensibilite, et ce qu'elle fait du bruit.

Chapitre 4, partie I, planches 10 a 20. En regime etabli, un capteur rend
`Y = f(X) + N` : la grandeur a mesurer `X`, transformee par `f`, plus un
bruit additif. Sa **sensibilite** en `X_0` est la pente `f'(X_0)`.

Elle dit ce que le bruit coute : l'estimation `f^-1(Y)` s'ecarte de `X`
d'environ `N / S`, par la regle de derivation de la fonction reciproque.
Un capteur peu sensible transforme un petit bruit en grande erreur.

L'anemometre a fil chaud, `I^2 = A' + B' U^n`, est l'exemple du cours
d'une caracteristique non lineaire : on l'inverse numeriquement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import sympy as sp


def sensibilite(
    caracteristique: sp.Expr, grandeur: sp.Symbol, point: object
) -> sp.Expr:
    """`S = df/dX` en `X = X_0`."""
    return sp.diff(caracteristique, grandeur).subs(grandeur, point)


def erreur_d_estimation(bruit: float, pente: float) -> float:
    """`dX ~ N / S` : l'ecart que le bruit additif `N` cause sur l'estimation."""
    if pente == 0:
        msg = "sensibilite nulle : la mesure ne dit rien de la grandeur"
        raise ValueError(msg)
    return bruit / pente


@dataclass(frozen=True, slots=True)
class FilChaud:
    """L'anemometre a temperature constante : `I^2 = A' + B' U^n`."""

    ordonnee: float
    pente: float
    exposant: float

    def __post_init__(self) -> None:
        """Refuse des coefficients qui ne decrivent pas un anemometre."""
        if self.ordonnee < 0 or self.pente <= 0 or self.exposant <= 0:
            msg = (
                f"coefficients invalides : A' = {self.ordonnee}, "
                f"B' = {self.pente}, n = {self.exposant}"
            )
            raise ValueError(msg)

    def courant(self, vitesse: float) -> float:
        """`I = racine(A' + B' U^n)`."""
        if vitesse < 0:
            msg = f"vitesse negative : {vitesse} m/s"
            raise ValueError(msg)
        puissance = math.pow(vitesse, self.exposant)
        return math.sqrt(self.ordonnee + self.pente * puissance)

    def vitesse(self, courant: float) -> float:
        """`U = ((I^2 - A') / B')^(1/n)` : l'inversion que l'unite numerique fait."""
        ecart = courant**2 - self.ordonnee
        if ecart < 0:
            msg = f"courant inferieur au courant a vitesse nulle : {courant} A"
            raise ValueError(msg)
        return math.pow(ecart / self.pente, 1 / self.exposant)

    def sensibilite(self, vitesse: float) -> float:
        """`dI/dU = n B' U^(n - 1) / (2 I)`."""
        puissance = math.pow(vitesse, self.exposant - 1)
        derivee = self.exposant * self.pente * puissance
        return derivee / (2 * self.courant(vitesse))
