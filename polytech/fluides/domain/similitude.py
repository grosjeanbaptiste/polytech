"""La similitude complete de deux ecoulements.

Planches du chapitre VII, en rappel du chapitre VI. Deux ecoulements
permanents et incompressibles de fluides reels sont en similitude
complete si leurs nombres de Reynolds et de Froude sont egaux, aux points
homologues. Les deux conditions ensemble imposent

`(nu_1 / nu_2)^2 (L_2 / L_1)^3 (g_2 / g_1) = 1` :

une maquette a echelle reduite demande un fluide moins visqueux, ce qui
rend la similitude complete rarement realisable.
"""

from __future__ import annotations

import math

from polytech.fluides.domain.grandeurs import PESANTEUR


def nombre_de_froude(
    vitesse: float, longueur: float, pesanteur: float = PESANTEUR
) -> float:
    """`Fr = U / racine(g L)`."""
    if longueur <= 0:
        msg = f"longueur nulle ou negative : {longueur} m"
        raise ValueError(msg)
    return vitesse / math.sqrt(pesanteur * longueur)


def rapport_des_viscosites_en_similitude_complete(
    *, echelle: float, rapport_des_pesanteurs: float = 1.0
) -> float:
    """`nu_2 / nu_1 = e^(3/2) racine(g_2 / g_1)`, `e` etant l'echelle `L_2 / L_1`."""
    if echelle <= 0:
        msg = f"echelle nulle ou negative : {echelle}"
        raise ValueError(msg)
    return math.sqrt(echelle**3 * rapport_des_pesanteurs)
