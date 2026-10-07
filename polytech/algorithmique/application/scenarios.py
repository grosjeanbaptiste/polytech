"""Une saisie par programme : le scenario sur lequel le corrige est eprouve.

Chaque scenario nomme le fichier C++, ce que l'utilisateur tape, et le
modele qui dit ce qui doit s'afficher. Les saisies sont choisies pour
passer par chaque point du menu et par les cas limites de l'enonce.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from polytech.algorithmique.application import (
    tp_conteneurs,
    tp_demarrage,
    tp_enregistrements,
    tp_heritage,
    tp_matieres,
    tp_objets,
    tp_tableaux,
)

if TYPE_CHECKING:
    from collections.abc import Callable


@dataclass(frozen=True, slots=True)
class Scenario:
    """Un programme, sa saisie, et le modele de ce qu'il affiche."""

    dossier: str
    source: str
    saisie: str
    modele: Callable[[str], tuple[str, ...]]


DE_2019: Final = "tp-01-a-06-2019"
DE_2017: Final = "tp-04-a-06-2017"
"""Les deux dossiers de travail, un par feuille de TP."""
SCENARIOS: Final = {
    "tp1_tp2": Scenario(
        DE_2019,
        "tp1_tp2.cpp",
        "0 11 5  1  1 2 3 4 5  6 7 7 5  3 9  4 2 8\n",
        tp_tableaux.derouler,
    ),
    "tp3": Scenario(
        DE_2019,
        "tp3.cpp",
        "1 5  zoe b  ali c  zoe b  bob a  ali c  2 3 4 8  7 zoe b  5 0  6 1 eve z"
        "  6 9 x y  9\n",
        tp_enregistrements.derouler,
    ),
    "tp4": Scenario(
        DE_2019,
        "tp4.cpp",
        "1 7 zoe  1 3 ali  4  3 7  2  5   1 max  1 lea  4  3 12  2  5\n",
        tp_objets.derouler,
    ),
    "tp4_dynamique": Scenario(
        DE_2019,
        "tp4_dynamique.cpp",
        "1 max  1 lea  3 1 max  3 9 max  2  1 bob  4\n",
        tp_objets.derouler_dynamique,
    ),
    "tp5": Scenario(
        DE_2019,
        "tp5.cpp",
        "1 2  7 zoe  3 ali  2 D 5 bob  4 5  3 3 ali  3 4 ali  0 P"
        "  1 2  2 curie physique  1 gauss maths  2 F 9 euler maths  4 5"
        "  3 1 gauss maths  3 99 gauss maths  3 8 gauss chimie  6\n",
        tp_conteneurs.derouler,
    ),
    "tp6": Scenario(
        DE_2019,
        "tp6.cpp",
        "1 noether algebre doyenne  1 4 zoe 12 14 16  1 5 ali 10 11 13"
        "  2 2 gauss maths  2 3 curie physique  4  5 ali  5 curie  3 E 0  3 P 5"
        "  4  6\n",
        tp_heritage.derouler,
    ),
    "tp6_matieres": Scenario(
        DE_2019,
        "tp6_matieres.cpp",
        "1 zoe 2 math 14 info 16  1 ali 4 chimie 10 info 12 mecara 8 anglais 15"
        "  1 bob 0  2 gauss math  2 turing info  3  4 turing  4 gauss  4 curie  5\n",
        tp_matieres.derouler,
    ),
    "tp4_demarrage": Scenario(
        DE_2017,
        "tp4_demarrage.cpp",
        "1 F 7 zoe  1 D 3 ali  1 x  4  3 7  2  5"
        "   1 D max  1 D lea  1 M bob  4  3 2  2  5\n",
        tp_demarrage.derouler,
    ),
}
"""Les blancs ne comptent pas : `cin >>` lit mot a mot, sur une ligne ou plusieurs."""
