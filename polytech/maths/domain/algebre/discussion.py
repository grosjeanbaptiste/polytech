"""Discuter un systeme lineaire carre selon la valeur d'un parametre.

Tant que le determinant de la matrice n'est pas nul, le systeme a une
solution et une seule, que donne la regle de Cramer. Il ne reste a
examiner que les valeurs du parametre qui l'annulent : pour chacune, le
systeme est soit indetermine, soit impossible, selon que le second membre
est ou non compatible.

Un determinant nul pour **toute** valeur du parametre n'a aucune valeur
singuliere a examiner : la methode ne s'applique pas, et `discuter` leve
au lieu de rendre un dictionnaire vide, qui se lirait « aucun cas
particulier ». C'est `discuter_par_le_rang` qui tranche alors.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.algebre.determinants import Determinant
from polytech.maths.domain.algebre.solutions import Nature, nature_des_solutions
from polytech.maths.domain.algebre.systemes import SystemeLineaire


def discuter(systeme: SystemeLineaire, parametre: sp.Symbol) -> dict[sp.Expr, Nature]:
    """La nature du systeme en chaque valeur du parametre qui annule le determinant."""
    if sp.expand(systeme.matrice.det()) == 0:
        msg = (
            f"determinant nul pour toute valeur de {parametre} : "
            "discuter_par_le_rang, la regle de Cramer n'a rien a examiner"
        )
        raise ValueError(msg)
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
