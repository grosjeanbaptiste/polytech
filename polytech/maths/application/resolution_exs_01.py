"""Mathematique, exercices 1 : la forme d'une resolution, et ses gestes communs.

Une primitive se rend avec ce qui y mene : la methode, les etats
intermediaires que le corrige montre, et la reponse. Le module porte
aussi le geste qui revient trois fois dans la feuille -- deux
integrations par parties qui ramenent l'integrale cherchee, et
l'equation qu'elle verifie alors.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import sympy as sp

from polytech.maths.domain.integration.parties import IntegrationParParties
from polytech.maths.domain.symboles import VARIABLE as X

A: Final = sp.Symbol("a", positive=True)
B: Final = sp.Symbol("b", positive=True)
"""Les parametres des exercices 7, 9 et 13, pris strictement positifs."""

PAR_EQUATION: Final = "deux integrations par parties, puis une equation"


@dataclass(frozen=True, slots=True)
class Resolution:
    """Un integrande, la methode, les etapes a montrer et la primitive."""

    integrande: sp.Expr
    methode: str
    etapes: tuple[sp.Expr, ...]
    primitive: sp.Expr


def par_equation(
    premier: tuple[sp.Expr, sp.Expr], second: tuple[sp.Expr, sp.Expr]
) -> Resolution:
    """`I = u_1 v_1 - (u_2 v_2 - k I)`, d'ou `I = (u_1 v_1 - u_2 v_2) / (1 - k)`.

    `premier` et `second` sont les couples `(u, dv)` des deux gestes. Le
    second porte sur l'integrale que le premier laisse ; celle qu'il laisse
    a son tour doit etre un multiple `k` de l'integrale de depart.
    """
    un, deux = IntegrationParParties(*premier), IntegrationParParties(*second)
    integrande = premier[0] * premier[1]
    restante = deux.v() * deux.derivee_de_u()
    rapport = sp.simplify(restante / integrande)
    if rapport.has(X):
        msg = f"le second geste ne ramene pas l'integrale de depart : {rapport}"
        raise ValueError(msg)
    primitive = (un.u * un.v() - deux.u * deux.v()) / (1 - rapport)
    etapes = (un.etape(), deux.etape())
    return Resolution(integrande, PAR_EQUATION, etapes, sp.simplify(primitive))
