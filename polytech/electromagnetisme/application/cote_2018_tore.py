"""Exercice cote 2018, question 1 : deux demi-tores de materiaux differents.

Le flux est le meme partout et la section aussi : l'induction vaut 1,1 T
dans l'hipersil, dans les toles et dans l'air. Seul le champ differe, et
c'est lui que le theoreme d'Ampere additionne.
"""

from __future__ import annotations

import math
from typing import Final

from polytech.electromagnetisme.domain.magnetostatique.circuits_magnetiques import (
    potentiel_scalaire_magnetique,
)
from polytech.electromagnetisme.domain.magnetostatique.entrefer import (
    CircuitAEntrefers,
    champ_dans_l_entrefer,
)

INDUCTION: Final = 1.1
"""`B`, en teslas : la valeur voulue dans l'hipersil, donc partout."""
CHAMP_LU_DANS_L_HIPERSIL: Final = 20.0
"""`H_H`, en A/m, **lu sur la planche B** (hipersil) pour 1,1 T."""
CHAMP_LU_DANS_LES_TOLES: Final = 380.0
"""`H_T`, en A/m, **lu sur la planche A** (toles ordinaires) pour 1,1 T."""
RAYON_INTERIEUR: Final = 10e-2
RAYON_EXTERIEUR: Final = 11e-2
SPIRES: Final = 2 * 500
"""Deux demi-enroulements de 500 spires, en serie et concordants."""
COURANT: Final = 1.0


def longueur_d_un_demi_tore() -> float:
    """`pi r`, sur la ligne moyenne `r = (R1 + R2) / 2`."""
    return math.pi * (RAYON_INTERIEUR + RAYON_EXTERIEUR) / 2


def champ_d_entrefer() -> float:
    """`H_e = B / mu_0`, le papier ayant `mu_r = 1`."""
    return champ_dans_l_entrefer(INDUCTION)


def partage() -> tuple[float, float, float]:
    """Les amperes-tours de l'hipersil, des toles, puis des deux entrefers."""
    longueur = longueur_d_un_demi_tore()
    hipersil = potentiel_scalaire_magnetique(CHAMP_LU_DANS_L_HIPERSIL, longueur)
    toles = potentiel_scalaire_magnetique(CHAMP_LU_DANS_LES_TOLES, longueur)
    return hipersil, toles, SPIRES * COURANT - hipersil - toles


def epaisseur() -> float:
    """`l_e`, en metres : ce qui reste des amperes-tours, sur deux entrefers."""
    return partage()[2] / (2 * champ_d_entrefer())


def circuit() -> CircuitAEntrefers:
    """Le meme circuit, reconstruit avec l'epaisseur trouvee.

    Le fer y est decrit par son champ moyen sur les deux demi-tores, de
    meme longueur : c'est ce que la somme `H_H l + H_T l` revient a ecrire.
    """
    moyen = (CHAMP_LU_DANS_L_HIPERSIL + CHAMP_LU_DANS_LES_TOLES) / 2
    return CircuitAEntrefers(
        champ_dans_le_fer=moyen,
        longueur_du_fer=2 * longueur_d_un_demi_tore(),
        induction=INDUCTION,
        epaisseur=epaisseur(),
        entrefers=2,
    )
