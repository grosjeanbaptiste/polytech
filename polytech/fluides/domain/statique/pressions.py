"""Les pressions absolue, relative et de vide, et les lois de Pascal et Torricelli.

Chapitre II des planches. La pression **absolue** se mesure depuis le
vide absolu ; la pression **relative** depuis une reference, en general
l'atmosphere : `p_gauge = p_abs - p_atm`. Sous l'atmosphere, on parle de
vide : `p_vac = p_atm - p_abs`.

Loi de Pascal : la pression est la meme sur un plan horizontal d'un meme
fluide, donc deux pistons relies supportent des forces proportionnelles a
leurs aires, `F_2 = F_1 A_2 / A_1` -- la presse hydraulique.

Barometre de Torricelli : la colonne de mercure que l'atmosphere
soutient mesure la pression absolue, `P_atm = rho g h`.
"""

from __future__ import annotations

from typing import Final

from polytech.fluides.domain.grandeurs import PESANTEUR, Hauteur, MasseVolumique

BAR: Final = 1e5
"""Un bar, en pascals."""

ATMOSPHERE: Final = 101_325.0
"""Une atmosphere normale, en pascals."""


def en_bars(pascals: float) -> float:
    """Une pression en pascals, lue en bars."""
    return pascals / BAR


def en_atmospheres(pascals: float) -> float:
    """Une pression en pascals, lue en atmospheres normales."""
    return pascals / ATMOSPHERE


def _exiger_l_absolue(absolue: float) -> float:
    if absolue < 0:
        msg = f"pression absolue negative : {absolue} Pa"
        raise ValueError(msg)
    return absolue


def pression_relative(absolue: float, reference: float) -> float:
    """`p_gauge = p_abs - p_atm`, en pascals : negative sous la reference."""
    return _exiger_l_absolue(absolue) - reference


def pression_absolue(relative: float, reference: float) -> float:
    """`p_abs = p_gauge + p_atm`, en pascals."""
    return _exiger_l_absolue(relative + reference)


def vide(absolue: float, reference: float) -> float:
    """`p_vac = p_atm - p_abs`, en pascals : defini sous la reference seulement."""
    if absolue > reference:
        msg = f"pas un vide : {absolue} Pa depasse la reference {reference} Pa"
        raise ValueError(msg)
    return reference - _exiger_l_absolue(absolue)


def force_de_la_presse(force: float, aire: float, autre_aire: float) -> float:
    """`F_2 = F_1 A_2 / A_1` : la force sur le second piston, en newtons."""
    if aire <= 0 or autre_aire <= 0:
        msg = f"aire nulle ou negative : {aire} m^2 et {autre_aire} m^2"
        raise ValueError(msg)
    return force * autre_aire / aire


def hauteur_barometrique(
    pression: float, fluide: MasseVolumique, pesanteur: float = PESANTEUR
) -> Hauteur:
    """`h = P_atm / (rho g)` : la colonne que cette pression soutient."""
    return Hauteur(metres=_exiger_l_absolue(pression) / (fluide.kg_par_m3 * pesanteur))
