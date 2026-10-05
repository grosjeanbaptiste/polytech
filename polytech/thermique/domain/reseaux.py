"""Les reseaux de resistances thermiques, en regime stationnaire.

Syllabus de thermique, tome 1, sections 0.6 et 0.8. Un phenomene
thermique stationnaire se represente par une conductance `K`, ou par la
resistance `R = 1 / K` ; pour une surface `S`, la conductance vaut `K S`.

Deux modes qui agissent en meme temps et independamment, comme la
convection et le rayonnement d'une meme surface, sont en parallele : les
conductances s'ajoutent. Des modes que la chaleur doit franchir l'un
apres l'autre sont en serie : les resistances s'ajoutent. Et l'ecart de
temperature est le flux fois la resistance, `Delta t = Phi R`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence


def _exiger_des_resistances(resistances: Sequence[float]) -> None:
    if not resistances:
        msg = "aucune resistance : le reseau est vide"
        raise ValueError(msg)
    if any(resistance <= 0 for resistance in resistances):
        msg = f"resistance nulle ou negative : {tuple(resistances)}"
        raise ValueError(msg)


def en_serie(*resistances: float) -> float:
    """La resistance d'une chaine : la somme."""
    _exiger_des_resistances(resistances)
    return sum(resistances)


def en_parallele(*resistances: float) -> float:
    """La resistance de branches paralleles : l'inverse de la somme des inverses."""
    _exiger_des_resistances(resistances)
    return 1 / sum(1 / resistance for resistance in resistances)


def echauffement(flux: float, resistances: Sequence[float]) -> float:
    """`Delta t = Phi R`, `R` etant la somme des resistances traversees."""
    return flux * en_serie(*resistances)
