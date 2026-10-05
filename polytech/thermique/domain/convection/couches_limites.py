"""Les couches limites dont depend la convection : plaque, tube, cylindre.

Syllabus de thermique, tome 2, premiere partie, section 1.3.2. Le
transfert se joue dans la mince couche de fluide ralentie par la paroi.

Sur une plaque, la couche limite laminaire s'epaissit en
`delta / x = 4,64 Re_x^-0,5` -- la solution integrale du profil cubique
--, jusqu'a la transition, pour `Re_x` de 3e5 a 5e5. Au-dela subsiste
une sous-couche visqueuse, `delta_l / x = 72,94 Re_x^-0,9`, qui porte
presque tout l'ecart de temperature.

Dans un tube, l'ecoulement est laminaire sous `Re_d = 2320`, et son
profil n'est etabli qu'apres `L_e = 0,03 Re_d d`.

Autour d'un cylindre, la couche limite decolle vers 82 degres tant
qu'elle est laminaire, vers 110 degres quand elle devient turbulente,
au-dela de `Re = 4e5` : le sillage se resserre et la trainee chute.
"""

from __future__ import annotations

from typing import Final

REYNOLDS_CRITIQUE_SUR_PLAQUE: Final = (3e5, 5e5)
"""La transition, selon la turbulence amont et la rugosite de la plaque."""
REYNOLDS_CRITIQUE_EN_TUBE: Final = 2320.0
"""Sous ce nombre, l'ecoulement etabli est celui de Hagen-Poiseuille."""
REYNOLDS_DE_LA_CHUTE_DE_TRAINEE: Final = 4e5
"""Au-dela, la couche limite du cylindre transite avant de decoller."""


def epaisseur_laminaire(distance: float, reynolds: float) -> float:
    """`delta = 4,64 x Re_x^-0,5`, en metres."""
    if reynolds >= REYNOLDS_CRITIQUE_SUR_PLAQUE[1]:
        msg = f"couche limite turbulente : Re_x = {reynolds:.3g}"
        raise ValueError(msg)
    return float(4.64 * distance * reynolds**-0.5)


def epaisseur_de_sous_couche(distance: float, reynolds: float) -> float:
    """`delta_l = 72,94 x Re_x^-0,9`, en metres."""
    if reynolds < REYNOLDS_CRITIQUE_SUR_PLAQUE[0]:
        msg = f"couche limite encore laminaire : Re_x = {reynolds:.3g}"
        raise ValueError(msg)
    return float(72.94 * distance * reynolds**-0.9)


def longueur_d_etablissement(diametre: float, reynolds: float) -> float:
    """`L_e = 0,03 Re_d d`, en regime laminaire."""
    if reynolds > REYNOLDS_CRITIQUE_EN_TUBE:
        msg = f"ecoulement turbulent : Re_d = {reynolds:.3g}"
        raise ValueError(msg)
    return 0.03 * reynolds * diametre


def angle_de_decollement(reynolds: float) -> float:
    """L'angle, compte depuis le point d'arret, ou la couche limite decolle."""
    if reynolds <= 0:
        msg = f"nombre de Reynolds nul ou negatif : {reynolds}"
        raise ValueError(msg)
    if reynolds > REYNOLDS_DE_LA_CHUTE_DE_TRAINEE:
        return 110.0
    return 82.0
