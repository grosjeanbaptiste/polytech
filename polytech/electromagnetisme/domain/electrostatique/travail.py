"""Le travail des forces du champ, et la tension electrique.

Le syllabus d'electricite generale (2.19) : deplacer une charge `Q` du
point 1 au point 2 fait travailler les forces du champ de

    T_12 = (V_1 - V_2) Q = Q U_12,       U_12 = V_1 - V_2   (2.19.5 a 2.19.7)

ou `U_12` est la difference de potentiel, ou tension. Le travail qu'il
faut fournir **contre** les forces du champ est l'oppose, `Q U_21`
(2.19.8). Ni l'un ni l'autre ne depend du chemin suivi.
"""

from __future__ import annotations

from typing import Any


def tension(premier: Any, second: Any) -> Any:  # noqa: ANN401 -- reels ou symboles
    """`U_12 = V_1 - V_2`."""
    return premier - second


def travail_des_forces_du_champ(charge: Any, depart: Any, arrivee: Any) -> Any:  # noqa: ANN401 -- reels ou symboles
    """`T_12 = Q (V_1 - V_2)` : positif si la charge positive descend les potentiels."""
    return charge * tension(depart, arrivee)


def travail_contre_le_champ(charge: Any, depart: Any, arrivee: Any) -> Any:  # noqa: ANN401 -- reels ou symboles
    """`Q U_21 = -T_12` : ce qu'il faut fournir pour amener la charge de 1 a 2."""
    return charge * tension(arrivee, depart)
