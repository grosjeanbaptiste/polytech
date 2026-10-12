"""L'etude de chaque matrice du chapitre X : spectre, nature, reduction.

Le procede du syllabus (X.2 a X.4) est dans le domaine ; ce module l'applique
a une matrice du recueil, dans le cas que l'on nomme, et met les vecteurs
propres **sans denominateur** : `(-3, 0, 2)` plutot que `(-3/2, 0, 1)`, et
surtout `(-a, 1 - a, a)` plutot que `(-1, (1 - a)/a, 1)`, qui n'existerait
pas pour `a = 0`.
"""

from __future__ import annotations

from functools import reduce

import sympy as sp

from polytech.maths.application import recueil_algebre_ch10_enonces as enonces
from polytech.maths.application import recueil_algebre_ch10_parametres as parametres
from polytech.maths.domain.algebre.reduction import (
    Nature,
    Reduction,
    nature_de_la_reduction,
    reduire_au_maximum,
)
from polytech.maths.domain.algebre.triangularisation import (
    completer_en_base,
    semblable_par,
)

GENERAL = ""
TROUVE = "Trouve"
"""Le cas ou les parametres ont la valeur que l'enonce fait trouver."""
Propre = tuple[sp.Expr, int, tuple[tuple[sp.Expr, ...], ...]]


def matrice(numero: int, cle: str = GENERAL) -> sp.Matrix:
    """La matrice de l'exercice, dans le cas nomme."""
    if numero not in enonces.MATRICES:
        msg = f"pas de matrice a l'exercice {numero} dans le chapitre X"
        raise KeyError(msg)
    if cle == GENERAL:
        return enonces.MATRICES[numero]
    if cle == TROUVE:
        return enonces.MATRICES[numero].subs(parametres.retenus(numero))
    if cle not in enonces.CAS.get(numero, {}):
        msg = f"exercice {numero} : pas de cas {cle!r}"
        raise KeyError(msg)
    return enonces.MATRICES[numero].subs(enonces.CAS[numero][cle])


def _multiple(colonne: sp.Matrix) -> sp.Expr:
    """Le plus petit multiple commun des denominateurs d'une colonne."""
    return reduce(sp.lcm, [sp.denom(sp.together(terme)) for terme in colonne])


def _sans_denominateur(colonne: sp.Matrix) -> tuple[sp.Expr, ...]:
    return tuple(sp.expand(sp.cancel(terme * _multiple(colonne))) for terme in colonne)


def caracteristique(numero: int, cle: str = GENERAL) -> sp.Expr:
    """`det(A - tI)`, factorise."""
    carree = matrice(numero, cle)
    return sp.factor((carree - enonces.T * sp.eye(carree.rows)).det())


def spectre(numero: int, cle: str = GENERAL) -> tuple[Propre, ...]:
    """Chaque valeur propre reelle, `m_a`, et une base de son sous-espace propre."""
    reels = [
        (valeur, algebrique, tuple(_sans_denominateur(v) for v in vecteurs))
        for valeur, algebrique, vecteurs in matrice(numero, cle).eigenvects()
        if valeur.is_real
    ]
    return tuple(sorted(reels, key=lambda propre: sp.default_sort_key(propre[0])))


def nature(numero: int, cle: str = GENERAL) -> Nature:
    """Diagonalisable, triangularisable seulement, ou ni l'un ni l'autre."""
    return nature_de_la_reduction(matrice(numero, cle))


def reduction(numero: int, cle: str = GENERAL) -> Reduction:
    """La reduction au maximum, les colonnes du passage sans denominateur."""
    carree = matrice(numero, cle)
    brute = reduire_au_maximum(carree)
    colonnes = [brute.passage[:, j] for j in range(brute.passage.cols)]
    passage = sp.Matrix.hstack(*[c * _multiple(c) for c in colonnes])
    passage = passage.applyfunc(lambda terme: sp.expand(sp.cancel(terme)))
    return Reduction(
        passage, semblable_par(matrice(numero, cle), passage), brute.etapes
    )


def premiere_etape(numero: int, cle: str = GENERAL) -> tuple[sp.Matrix, sp.Matrix]:
    """`P_1`, les vecteurs propres prolonges en une base, et `P_1^-1 A P_1`."""
    propres = [sp.Matrix(v) for _, _, base in spectre(numero, cle) for v in base]
    passage = completer_en_base(propres)
    return passage, semblable_par(matrice(numero, cle), passage)
