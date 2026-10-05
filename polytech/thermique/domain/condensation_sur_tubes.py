"""La condensation sur des tubes horizontaux, et le Reynolds du film.

Syllabus de thermique, tome 2, premiere partie, sections 5.5 et 5.6.
Autour d'un tube horizontal de diametre `d`, le meme bilan que sur la
paroi verticale donne `K_h = 0,725 [...]^1/4` : a longueur egale, le
tube couche l'emporte sur le tube debout dans le rapport
`0,77 (H / d)^1/4`.

Dans un condenseur, le condensat d'un tube ruisselle sur ceux qui sont
sous lui et y epaissit le film : pour `n` tubes d'un meme plan vertical,
le coefficient moyen porte `n^-1/4`. Exprime par le debit `G_d` recueilli
sous la colonne, il devient `0,954 [...]^1/3`.

Le film a un nombre de Reynolds, `4 G / mu`, forme sur son diametre
hydraulique `4 delta`. Les coefficients s'ecrivent alors sous forme
reduite : `1,47 Re^-1/3` pour la paroi verticale, `1,2 Re_D^-1/3` pour
les tubes horizontaux, et `0,0077 Re^0,4` au-dela de la transition, que
l'on observe vers `Re = 1800`.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from polytech.thermique.domain.film_de_condensat import Condensat

REYNOLDS_CRITIQUE_DU_FILM: Final = 1800.0
"""Page 74 : la transition du film vers la turbulence."""
NUSSELT_HORIZONTAL: Final = 0.725
"""(5.13) : le coefficient du tube horizontal, 0,728 par l'integrale exacte."""
_TIERS = 1.0 / 3.0


def coefficient_horizontal(
    condensat: Condensat, diametre: float, ecart: float, *, tubes: int = 1
) -> float:
    """`K_h = 0,725 [l_v X / (n d (t_sat - t_p))]^1/4`, `X` etant le groupe du film."""
    moteur = condensat.chaleur_latente * condensat.groupe()
    return float(NUSSELT_HORIZONTAL * (moteur / (tubes * diametre * ecart)) ** 0.25)


def coefficient_horizontal_par_le_debit(condensat: Condensat, debit: float) -> float:
    """`K_h = 0,954 [(rho_l - rho_v) rho_l g lambda^3 / (mu G_d)]^1/3`."""
    constante = (NUSSELT_HORIZONTAL**4 * math.pi) ** _TIERS
    return float(constante * (condensat.groupe() / debit) ** _TIERS)


def rapport_horizontal_sur_vertical(hauteur: float, diametre: float) -> float:
    """`K_h / K_v = 0,77 (H / d)^1/4`, pour un meme tube couche ou debout."""
    vertical = 4 / 3 * 0.25**0.25
    return float(NUSSELT_HORIZONTAL / vertical * (hauteur / diametre) ** 0.25)


def reynolds_du_film(debit: float, viscosite: float) -> float:
    """`Re = 4 G / mu`."""
    if viscosite <= 0:
        msg = f"viscosite nulle ou negative : {viscosite} Pa s"
        raise ValueError(msg)
    return 4 * debit / viscosite


def reynolds_sous_une_colonne(debit: float, viscosite: float) -> float:
    """`Re_D = 2 G_D / mu` : le debit se partage de part et d'autre du tube."""
    return reynolds_du_film(debit / 2, viscosite)


def coefficient_vertical_par_le_reynolds(
    condensat: Condensat, reynolds: float
) -> float:
    """`1,47 Re^-1/3` en laminaire, `0,0077 Re^0,4` au-dela de 1800, sur l'echelle."""
    if reynolds > REYNOLDS_CRITIQUE_DU_FILM:
        return float(0.0077 * reynolds**0.4 / condensat.echelle())
    return float(1.47 * reynolds**-_TIERS / condensat.echelle())
