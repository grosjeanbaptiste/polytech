"""Les ecrans thermiques entre deux plans, et la canne pyrometrique.

Syllabus de thermique, tome 1, deuxieme partie, section 3.7. Une feuille
mince glissee entre deux plans paralleles prend la temperature qui
egalise ce qu'elle recoit et ce qu'elle cede. Si elle est de meme nature
que les plans, `T_3^4` est la moyenne des deux autres, et le flux tombe
de moitie ; `n` ecrans le divisent par `n + 1`. Un ecran peu emissif
fait mieux : chaque intervalle oppose `1/eps + 1/eps_3 - 1`.

Le meme bilan explique l'erreur d'une canne pyrometrique. Le
thermocouple recoit du gaz par convection et cede aux parois par
rayonnement ; il lit donc une temperature comprise entre les deux, que
seul un ecran rapproche de celle du gaz.
"""

from __future__ import annotations

from scipy.optimize import brentq

from polytech.thermique.domain.rayonnement.corps_noir import emittance
from polytech.thermique.domain.rayonnement.surfaces_grises import (
    flux_entre_deux_plans_gris,
)


def temperature_de_l_ecran(chaude: float, froide: float) -> float:
    """`T_3`, telle que `T_3^4 = (T_1^4 + T_2^4) / 2`."""
    return float(((chaude**4 + froide**4) / 2) ** 0.25)


def flux_a_travers_des_ecrans(
    emissivite: float, chaude: float, froide: float, *, ecrans: int
) -> float:
    """Le flux par metre carre, a travers `ecrans` feuilles de meme nature."""
    if ecrans < 0:
        msg = f"nombre d'ecrans negatif : {ecrans}"
        raise ValueError(msg)
    sans = flux_entre_deux_plans_gris(emissivite, emissivite, chaude, froide, 1.0)
    return sans / (ecrans + 1)


def flux_avec_un_ecran_different(
    emissivite: float, ecran: float, chaude: float, froide: float
) -> float:
    """Le flux par metre carre, a travers un ecran d'emissivite `ecran`."""
    intervalle = flux_entre_deux_plans_gris(emissivite, ecran, chaude, froide, 1.0)
    return intervalle / 2


def temperature_lue_par_la_canne(
    gaz: float, paroi: float, *, convection: float, emissivite: float
) -> float:
    """`T_c`, telle que `K (T_g - T_c) = eps sigma (T_c^4 - T_p^4)`.

    La canne, petite devant le conduit, a pour facteur corrige son
    emissivite.
    """

    def bilan(canne: float) -> float:
        rayonne = emissivite * (emittance(canne) - emittance(paroi))
        return convection * (gaz - canne) - rayonne

    return float(brentq(bilan, min(gaz, paroi), max(gaz, paroi)))
