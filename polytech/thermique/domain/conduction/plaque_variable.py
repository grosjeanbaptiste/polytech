"""La plaque plongee dans un fluide : la solution en serie.

Syllabus de thermique, tome 1, premiere partie, section 4.2. Une plaque
d'epaisseur `2 l`, uniforme a `t_0`, est plongee dans un fluide a `t_f`.
L'ecart `theta = t - t_f` obeit a `d(theta)/dtau = a d2(theta)/dx2`, avec
la symetrie au centre et la condition de Fourier en `x = l`. La methode
de factorisation donne

`theta / theta_0 = somme A_n exp(-delta_n^2 Fo) cos(delta_n x / l)`,

les `delta_n` etant les racines de `cot(delta) = delta / Bi` et
`A_n = 2 sin(delta_n) / (delta_n + sin(delta_n) cos(delta_n))`. Tout se
dit avec deux nombres : Fourier, `Fo = a tau / l^2`, le temps rapporte a
la constante `l^2 / a` ; Biot, `Bi = K l / lambda`, le rapport de la
resistance de conduction a celle de surface.

Les termes d'ordre eleve s'eteignent vite : il ne reste bientot que le
fondamental. Un parallelepipede est le produit de trois plaques.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from scipy.optimize import brentq

if TYPE_CHECKING:
    from collections.abc import Sequence

MARGE = 1e-12
"""Pour rester a l'interieur de la branche de la cotangente."""


@dataclass(frozen=True, slots=True)
class PlaqueEnEchelon:
    """Une plaque sous un echelon de la temperature du fluide, par son Biot."""

    biot: float

    def __post_init__(self) -> None:
        """Refuse une plaque isolee : la serie n'aurait pas de valeur propre."""
        if self.biot <= 0:
            msg = f"nombre de Biot nul ou negatif : {self.biot}"
            raise ValueError(msg)

    def valeurs_propres(self, nombre: int) -> list[float]:
        """Les `nombre` premieres racines de `delta tan(delta) = Bi`."""
        racines = []
        for rang in range(nombre):
            debut = rang * math.pi + MARGE
            fin = rang * math.pi + math.pi / 2 - MARGE
            racines.append(brentq(lambda d: d * math.tan(d) - self.biot, debut, fin))
        return racines

    def amplitude(self, racine: float) -> float:
        """`A_n / theta_0 = 2 sin(d) / (d + sin(d) cos(d))`."""
        sinus = math.sin(racine)
        return 2 * sinus / (racine + sinus * math.cos(racine))

    def ecart_reduit(self, fourier: float, abscisse: float, termes: int = 60) -> float:
        """`theta / theta_0` au temps `Fo` et a l'abscisse reduite `x / l`."""
        return sum(
            self.amplitude(d) * math.exp(-(d**2) * fourier) * math.cos(d * abscisse)
            for d in self.valeurs_propres(termes)
        )


def produit_des_plaques(
    plaques: Sequence[PlaqueEnEchelon], points: Sequence[tuple[float, float]]
) -> float:
    """`theta / theta_0` d'un parallelepipede : le produit de ses plaques."""
    paires = zip(plaques, points, strict=True)
    return math.prod(plaque.ecart_reduit(*point) for plaque, point in paires)
