"""Les materiaux pour aimants permanents.

Le syllabus d'electricite generale (12.3), tableau 12.3.1 et texte de la
page 12-5, en unites SI. Le tableau ne donne que des ordres de grandeur
(« dizaines », « dixieme ») : ils sont portes ici par des intervalles, et
cette lecture est celle de ce module.

Le tableau imprime, pour le champ coercitif des ferrites, « 1 ou 2
centiemes » de kA/m. C'est impossible : le rectangle `(HB)max` est
inscrit dans la courbe de desaimantation, donc borne par `B_r H_c`, et
vingt amperes par metre ne laisseraient pas vingt joules par metre cube.
La note du tableau parle de 300 a 500 kA/m pour certains ferrites : il
faut lire des **centaines**, et c'est ce qui est retenu.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

Intervalle = tuple[float, float]

COERCITIF_IMPRIME_DES_FERRITES: Final[Intervalle] = (10.0, 20.0)
"""En A/m : « 1 ou 2 centiemes » de kA/m, tel que le tableau l'imprime."""


def borne_du_produit_d_energie(remanente: float, coercitif: float) -> float:
    """`B_r H_c`, en J/m^3 : aucun rectangle inscrit ne depasse ses deux cotes."""
    if remanente < 0 or coercitif < 0:
        msg = f"specification negative : B_r = {remanente} T, H_c = {coercitif} A/m"
        raise ValueError(msg)
    return remanente * coercitif


@dataclass(frozen=True, slots=True)
class MateriauPourAimant:
    """Les trois specifications d'un materiau, chacune par un intervalle."""

    produit: Intervalle
    remanente: Intervalle
    coercitif: Intervalle

    def est_coherent(self) -> bool:
        """Vrai si le plus petit `(HB)max` tient sous la plus grande borne."""
        plafond = borne_du_produit_d_energie(self.remanente[1], self.coercitif[1])
        return self.produit[0] <= plafond


MATERIAUX: Final = {
    "Alnico": MateriauPourAimant((10e3, 100e3), (1.0, 1.0), (10e3, 1000e3)),
    "ferrites": MateriauPourAimant((10e3, 20e3), (0.1, 1.0), (100e3, 200e3)),
    "samarium-cobalt": MateriauPourAimant((150e3, 250e3), (1.0, 1.0), (1000e3, 1000e3)),
}
"""Alnico et ferrites du tableau 12.3.1 ; SmCo5 et Sm2Co17 du texte qui suit."""
