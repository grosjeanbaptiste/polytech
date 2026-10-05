"""Discuter un systeme lineaire carre selon la valeur d'un parametre.

Tant que le determinant de la matrice n'est pas nul, le systeme a une
solution et une seule, que donne la regle de Cramer. Il ne reste a
examiner que les valeurs du parametre qui l'annulent : pour chacune, le
systeme est soit indetermine, soit impossible, selon que le second membre
est ou non compatible.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from polytech.maths.domain.algebre.determinants import Determinant
from polytech.maths.domain.algebre.solutions import Nature, nature_des_solutions
from polytech.maths.domain.algebre.systemes import SystemeLineaire

if TYPE_CHECKING:
    import sympy as sp


def discuter(systeme: SystemeLineaire, parametre: sp.Symbol) -> dict[sp.Expr, Nature]:
    """La nature du systeme en chaque valeur du parametre qui annule le determinant."""
    singulieres = Determinant(systeme.matrice).valeurs_singulieres(parametre)
    return {
        valeur: nature_des_solutions(
            SystemeLineaire(
                systeme.matrice.subs(parametre, valeur),
                systeme.second_membre.subs(parametre, valeur),
            )
        )
        for valeur in singulieres
    }
