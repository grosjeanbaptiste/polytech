"""Le gabarit d'un filtre non ideal, et l'ordre qu'il impose.

Un filtre ideal n'existe pas : le fascicule part donc d'un **gabarit** --
l'affaiblissement maximal tolere dans la bande passante, l'affaiblissement
minimal exige dans la bande bloquante, et les deux pulsations qui les
separent. De ces quatre nombres se deduit l'ordre, c'est-a-dire le nombre
de poles, c'est-a-dire le cout du filtre.

Les deux formules sont fermees et le cours les donne. Celle de Butterworth
emploie un logarithme decimal, celle de Chebyshev un argument cosinus
hyperbolique -- et c'est cette difference qui fait que Chebyshev demande
toujours moins de poles, au prix d'une ondulation dans la bande.

L'arrondi se fait **vers le haut**, toujours. Un ordre de 6,64 arrondi a 6
ne tiendrait pas le gabarit, et c'est precisement ce que le calcul sert a
eviter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

DECIBELS = 10.0
"""Les affaiblissements sont des rapports de puissance : 10 log, pas 20."""


@dataclass(frozen=True, slots=True)
class Gabarit:
    """Les quatre nombres qui specifient un filtre passe-bas."""

    affaiblissement_en_bande: float
    affaiblissement_hors_bande: float
    pulsation_passante: float
    pulsation_bloquante: float

    def __post_init__(self) -> None:
        """Refuse un gabarit qui decrirait un filtre impossible."""
        if self.pulsation_bloquante <= self.pulsation_passante:
            msg = (
                f"bande bloquante avant la bande passante : "
                f"{self.pulsation_bloquante} <= {self.pulsation_passante} rad/s"
            )
            raise ValueError(msg)
        if self.affaiblissement_hors_bande <= self.affaiblissement_en_bande:
            hors, dedans = (
                self.affaiblissement_hors_bande,
                self.affaiblissement_en_bande,
            )
            msg = f"affaiblissement hors bande trop faible : {hors} <= {dedans} dB"
            raise ValueError(msg)
        if self.pulsation_passante <= 0:
            msg = f"pulsation passante nulle ou negative : {self.pulsation_passante}"
            raise ValueError(msg)

    def selectivite(self) -> float:
        """`omega_s / omega_p` : plus il est proche de un, plus le filtre est cher."""
        return self.pulsation_bloquante / self.pulsation_passante

    def ecarts_en_puissance(self) -> tuple[float, float]:
        """Les deux quantites `10^(A/10) - 1` dont les deux formules partent.

        Publique, et non privee : les fonctions d'ordre vivent hors de la
        classe parce qu'elles dependent de la **famille** et non du
        gabarit, et un acces prive depuis l'exterieur serait une faute que
        nul `noqa` ne rachete.
        """
        return (
            10 ** (self.affaiblissement_hors_bande / DECIBELS) - 1,
            10 ** (self.affaiblissement_en_bande / DECIBELS) - 1,
        )


def ordre_de_butterworth(gabarit: Gabarit) -> int:
    """Le nombre de poles qu'un Butterworth exige pour tenir ce gabarit."""
    hors_bande, en_bande = gabarit.ecarts_en_puissance()
    numerateur = math.log10(hors_bande / en_bande)
    return max(1, math.ceil(numerateur / (2 * math.log10(gabarit.selectivite()))))


def ordre_de_chebyshev(gabarit: Gabarit) -> int:
    """Le nombre de poles qu'un Chebyshev de type I exige pour le meme gabarit.

    Toujours inferieur ou egal a celui de Butterworth : c'est la raison
    d'etre de la famille. Le prix est l'ondulation dans la bande, que le
    parametre `A_max` fixe directement.
    """
    hors_bande, en_bande = gabarit.ecarts_en_puissance()
    numerateur = math.acosh(math.sqrt(hors_bande / en_bande))
    return max(1, math.ceil(numerateur / math.acosh(gabarit.selectivite())))
