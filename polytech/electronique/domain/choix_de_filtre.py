"""Assembler et choisir un filtre (chapitre 3, partie II, planches 62 a 66).

Deux regles de mise en cascade, et la carte des technologies selon la
frequence -- des ordres de grandeur que la planche 66 dit elle-meme
« tres, tres grossiers ».
"""

from __future__ import annotations

import enum
import math
from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence


class Section(enum.IntEnum):
    """Le type d'une section, range dans l'ordre ou la cascade la place."""

    PASSE_BAS = 0
    PASSE_BANDE = 1
    PASSE_HAUT = 2


def ordonner_les_sections(sections: Sequence[Section]) -> tuple[Section, ...]:
    """Planche 64 : le passe-bas d'abord, le passe-haut en dernier.

    Le passe-bas tue les hautes frequences avant qu'elles ne soient
    amplifiees ; le passe-haut, en bout de chaine, tue le bruit en `1/f`
    que les etages precedents ont engendre.
    """
    if not sections:
        msg = "aucune section a ordonner"
        raise ValueError(msg)
    return tuple(sorted(sections))


class Technologie(enum.Enum):
    """Les familles de filtres de la planche 66."""

    CAPACITES_COMMUTEES = "capacites commutees"
    ACTIF_A_AOP = "actif a AOP (Sallen-Key, KHN, Tow-Thomas)"
    TRANSCONDUCTANCE = "a amplificateurs de transconductance"
    ECHELLE_LC = "echelle LC a composants discrets"
    MICRORUBAN = "microruban"


_PLAGES: Final = (
    (Technologie.CAPACITES_COMMUTEES, 0.0, 100e3),
    (Technologie.ACTIF_A_AOP, 0.0, 10e6),
    (Technologie.TRANSCONDUCTANCE, 10e6, 500e6),
    # « quelques GHz » : la borne est prise a 10 GHz, faute d'un nombre imprime.
    (Technologie.ECHELLE_LC, 10e6, 10e9),
    (Technologie.MICRORUBAN, 1e9, math.inf),
)


def technologies_possibles(frequence: float) -> tuple[Technologie, ...]:
    """Les familles dont la plage contient cette frequence, bornes hautes exclues."""
    if frequence < 0:
        msg = f"frequence negative : {frequence} Hz"
        raise ValueError(msg)
    return tuple(t for t, bas, haut in _PLAGES if bas <= frequence < haut)
