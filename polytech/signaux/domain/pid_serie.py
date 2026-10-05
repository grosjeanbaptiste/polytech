"""Le controleur PID sous sa forme serie, et ce qu'on lui demande.

Planches « Controleur » du cours de signaux et systemes. Plus de neuf
controleurs industriels sur dix sont des PID :

`C(p) = G_c (1 + p T_i) / (p T_i) (1 + p T_d) / (1 + p lambda T_d)`,

`G_c` etant le facteur de proportionnalite, `T_i` et `T_d` les constantes
de temps d'integration et de derivation, `lambda < 1` et
`alpha = T_i / T_d > 1`. Sans derivee il reste un PI ; l'industrie donne
`G_c` par la bande proportionnelle `a`, en pour cent, `G_c = 100 / a`.

Son module prend toutes les valeurs, mais son argument reste entre
-90 degres et un maximum que `alpha` et `lambda` fixent seuls.

Les specifications temporelles se traduisent en contraintes sur `|M|` :
`|M|max - |M(0)| < (D - 9) / 6` en dB, et `BP > 0,38 / t_m` en hertz.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import sympy as sp

from polytech.signaux.domain.symboles import IMAGE as P

REDUITES = np.logspace(-3, 5, 80_001)
"""Les pulsations reduites `w T_i` ou l'argument est cherche."""


FILTRE_USUEL = sp.Rational(1, 10)
"""Le coefficient du filtre, que le cours impose a 0,1."""


@dataclass(frozen=True, slots=True)
class PidSerie:
    """`G_c`, `T_i`, `T_d` et le coefficient `lambda` du filtre de la derivee."""

    gain: sp.Expr
    integrale: sp.Expr
    derivee: sp.Expr
    filtre: sp.Expr = FILTRE_USUEL

    def __post_init__(self) -> None:
        """Refuse une integration sans constante de temps."""
        if self.integrale == 0:
            msg = "constante d'integration nulle : le terme integral n'est pas defini"
            raise ValueError(msg)

    @classmethod
    def depuis_la_bande_proportionnelle(
        cls, bande: sp.Expr, integrale: sp.Expr, derivee: sp.Expr
    ) -> PidSerie:
        """`G_c = 100 / a`, la bande proportionnelle `a` etant en pour cent."""
        return cls(sp.Integer(100) / bande, integrale, derivee)

    def alpha(self) -> sp.Expr:
        """`alpha = T_i / T_d`."""
        return self.integrale / self.derivee

    def transmittance(self) -> sp.Expr:
        """`C(p)`."""
        integre = (1 + P * self.integrale) / (P * self.integrale)
        avance = (1 + P * self.derivee) / (1 + P * self.filtre * self.derivee)
        return self.gain * integre * avance

    def argument(self, reduite: float) -> float:
        """L'argument, en degres, a la pulsation reduite `w T_i`."""
        alpha, filtre = float(self.alpha()), float(self.filtre)
        avance = math.atan(reduite / alpha) - math.atan(filtre * reduite / alpha)
        return math.degrees(math.atan(reduite) - math.pi / 2 + avance)

    def avance_maximale(self) -> float:
        """Le plus grand argument que le controleur puisse apporter, en degres."""
        return max(self.argument(float(reduite)) for reduite in REDUITES[::40])


def ecart_maximal_de_m(depassement: float) -> float:
    """`|M|max - |M(0)|` admissible, en dB, pour un depassement en pour cent."""
    return max(0.0, (depassement - 9) / 6)


def bande_passante_minimale(temps_de_montee: float) -> float:
    """`BP > 0,38 / t_m`, en hertz."""
    if temps_de_montee <= 0:
        msg = f"temps de montee nul ou negatif : {temps_de_montee} s"
        raise ValueError(msg)
    return 0.38 / temps_de_montee
