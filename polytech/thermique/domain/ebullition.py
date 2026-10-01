"""L'ebullition : ses regimes, et la crise qui les separe.

Chapitre 6 du tome 2 du syllabus, et meme raison d'absence que les
precedents : la mesure de couverture ne regardait que les planches.

Le syllabus ouvre en disant que « le mecanisme en est tres compliqué et
imparfaitement connu a l'heure actuelle », et il motive le chapitre par
deux applications ou le flux est tel qu'aucun autre mode n'y suffit : le
refroidissement des reacteurs a eau, et « la limitation des echauffements
des circuits electroniques a haute densite (VLSI), ou des densites de flux
de 100000 W/m^2 sont depassees ».

Ce module porte donc ce que le chapitre etablit **sans abaque** : l'ordre
des regimes, et la distinction de pilotage qui decide du danger.

Les seuils de surchauffe entre regimes, eux, ne sont donnes qu'en figure
dans le syllabus -- la courbe de Nukiyama. Ils ne sont pas interpoles
ici : ce serait inventer une correlation, ce que ce depot refuse ailleurs
pour la lame de fluide entre 1400 et 6000.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Final

REGIMES_EN_CONVECTION_NATURELLE: Final = (
    "convection naturelle",
    "ebullition nucleee",
    "crise d'ebullition",
    "ebullition en film",
)
"""Les quatre regimes, dans l'ordre des surchauffes croissantes.

L'ordre **est** le contenu : la crise n'est pas un etat mais un
**passage**, et c'est pourquoi elle est entre deux regimes et non a une
extremite. Les intervertir perdrait exactement ce point.
"""

FLUX_DES_CIRCUITS_DENSES: Final = 1e5
"""`100000 W/m^2`, le flux que le syllabus dit depasse en VLSI.

C'est le chiffre qui justifie le chapitre : a ce niveau ni la convection
naturelle ni la convection forcee en gaz ne suffisent, et le changement
de phase devient le seul recours.
"""


class Pilotage(Enum):
    """Ce qu'on impose a la surface : son flux, ou sa temperature."""

    FLUX_IMPOSE = auto()
    TEMPERATURE_IMPOSEE = auto()


def risque_de_burn_out(pilotage: Pilotage, flux: float, critique: float) -> bool:
    """Vrai si depasser le flux critique detruit la surface.

    C'est le **pilotage** qui decide, et non le flux seul -- le point que
    le chapitre porte, et la raison pour laquelle la meme courbe se lit
    de deux facons.

    A flux impose -- un element electrique, un crayon de reacteur -- la
    paroi doit evacuer ce qu'on lui donne ; passe le flux critique elle
    ne peut plus le faire qu'en montant brutalement en temperature, d'ou
    le nom de « burn-out ». A temperature imposee -- un bain, un
    condenseur -- le meme depassement fait seulement chuter le flux, et
    rien ne brule.
    """
    if critique <= 0:
        msg = f"flux critique nul ou negatif : {critique} W/m^2"
        raise ValueError(msg)
    return pilotage is Pilotage.FLUX_IMPOSE and flux > critique
