"""Les piles et les accumulateurs : des chaines dont l'energie est chimique.

Une **pile hydro-electrique** est une chaine mixte, faite d'electrodes
metalliques plongees dans une solution electrolytique ; sa force
electromotrice vient des reactions aux contacts. Un **accumulateur** est
une pile reversible : les matieres actives transformees quand il debite
sont regenerees par un courant de sens inverse.

Des elements associes en serie additionnent leurs forces electromotrices :
c'est l'empilement de rondelles de zinc, de drap acidule et de cuivre que
Volta realisa en 1800, et qui a donne son nom a la pile.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
paragraphe 6.9, pages 6-9 et 6-10. Les chaines et les forces
electromotrices sont celles que le syllabus imprime.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True, slots=True)
class Pile:
    """Une pile : sa chaine, sa force electromotrice a vide, et ce qu'elle permet."""

    chaine: str
    fem: float
    reversible: bool = False
    etalon: bool = False


PILES: Final = {
    "Volta": Pile("Zn/H2SO4 aq/Cu", 0.85),
    "Leclanche": Pile("Zn/NH4Cl aq/MnO2", 1.5),
    "mercure": Pile("Zn/KOH aq/HgO", 1.35),
    "Weston": Pile("Hg.Cd/CdSO4 aq/HgSO4 aq/Hg", 1.0186, etalon=True),
}
"""Les quatre piles du syllabus ; la Weston vaut 1,018 6 V a 20 degC."""

ACCUMULATEURS: Final = {
    "cadmium-nickel": Pile("Cd/H2O/NiOOH", 1.3, reversible=True),
    "plomb": Pile("Pb/H2SO4/PbO2", 2.0, reversible=True),
}
"""Les deux accumulateurs du syllabus."""


def fem_d_un_empilement(pile: Pile, elements: int) -> float:
    """La force electromotrice de `elements` piles identiques en serie."""
    if elements < 1:
        msg = f"au moins un element attendu dans l'empilement, et non {elements}"
        raise ValueError(msg)
    return elements * pile.fem
