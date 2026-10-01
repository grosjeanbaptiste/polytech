"""Les echangeurs, par la moyenne logarithmique et par l'efficacite.

Ancrage direct sur le corrige du TP 4, dont trois valeurs se recoupent
entre elles :

* `K_gl S = 775,61 W/degC` et `Delta T_log = 15,582 degC` donnent
  `phi = 12,084 kW`, qui est bien la valeur publiee ;
* `K_gl S / S_i = 775,61 / 1,1556 = 671,2`, qui est bien le `K_gli`
  publie.

La garde sur les ecarts de meme signe n'est pas formelle : si les deux
ecarts terminaux sont de signes opposes, les courbes se croisent, ce qui
est thermodynamiquement impossible dans un echangeur -- et le logarithme
d'un rapport negatif ferait tomber le calcul bien plus loin, sur une
erreur de domaine qui ne dirait pas d'ou elle vient.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

RAPPORT_PROCHE_DE_UN = 1e-9
"""En dessous, la moyenne logarithmique degenere en moyenne arithmetique."""


def moyenne_logarithmique(ecart_a: float, ecart_b: float) -> float:
    """`(DT_A - DT_B) / ln(DT_A / DT_B)`, la moyenne du cours.

    Quand les deux ecarts sont egaux, la formule est indeterminee et sa
    limite est l'ecart lui-meme. Le cas se traite, il ne se contourne pas.
    """
    if ecart_a * ecart_b <= 0:
        msg = (
            f"ecarts terminaux de signes opposes ou nuls : {ecart_a} et "
            f"{ecart_b} -- les courbes de temperature se croiseraient"
        )
        raise ValueError(msg)
    if abs(ecart_a - ecart_b) <= RAPPORT_PROCHE_DE_UN * abs(ecart_a):
        return ecart_a
    return (ecart_a - ecart_b) / math.log(ecart_a / ecart_b)


@dataclass(frozen=True, slots=True)
class Echangeur:
    """Un echangeur decrit par sa conductance globale `K_gl S`."""

    conductance: float

    def __post_init__(self) -> None:
        """Refuse une conductance nulle : l'echangeur n'echangerait rien."""
        if self.conductance <= 0:
            msg = f"conductance globale nulle ou negative : {self.conductance}"
            raise ValueError(msg)

    def flux(self, ecart_a: float, ecart_b: float) -> float:
        """`phi = K_gl S Delta T_log`."""
        return self.conductance * moyenne_logarithmique(ecart_a, ecart_b)

    def coefficient_rapporte(self, surface: float) -> float:
        """`K_gl = (K_gl S) / S`, le coefficient par metre carre."""
        if surface <= 0:
            msg = f"surface nulle ou negative : {surface} m^2"
            raise ValueError(msg)
        return self.conductance / surface

    def nombre_d_unites_de_transfert(self, capacite_minimale: float) -> float:
        """`NUT = K_gl S / (q c)_min`, le nombre adimensionnel du cours."""
        if capacite_minimale <= 0:
            msg = (
                f"capacite calorifique minimale nulle ou negative : {capacite_minimale}"
            )
            raise ValueError(msg)
        return self.conductance / capacite_minimale


def efficacite_courants_croises_sans_melange(nut: float, rapport: float) -> float:
    """L'efficacite pour un echangeur a courants croises, fluides non melanges.

    Correlation classique `eps = 1 - exp(NUT^0,22 (exp(-R NUT^0,78) - 1) / R)`.
    Le formulaire la donne en abaque ; l'expression analytique vient donc
    de la litterature et non du cours -- c'est un apport exterieur, et il
    est signale comme tel.
    """
    if nut <= 0:
        msg = f"NUT nul ou negatif : {nut}"
        raise ValueError(msg)
    if not 0 < rapport <= 1:
        msg = f"rapport de capacites hors de ]0, 1] : {rapport}"
        raise ValueError(msg)
    exposant = nut**0.22 * (math.exp(-rapport * nut**0.78) - 1) / rapport
    return 1 - math.exp(exposant)
