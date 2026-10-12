"""Recueil d'algebre, chapitre III : resoudre et discuter les vingt-quatre systemes.

Les enonces sont dans `recueil_algebre_ch03_enonces`. Un systeme sans
parametre se resout par echelonnement simple ; un systeme a parametres se
discute sur les racines de son determinant, et chaque cas singulier se lit
sur la forme echelonnee de sa matrice augmentee.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.application import recueil_algebre_ch03_enonces as enonces
from polytech.maths.domain.algebre.echelonnement import echelonner
from polytech.maths.domain.algebre.resolution import par_echelonnement_simple
from polytech.maths.domain.algebre.solutions import (
    PARAMETRES,
    Nature,
    nature_des_solutions,
    solution_generale,
)
from polytech.maths.domain.algebre.systemes import SystemeLineaire

if TYPE_CHECKING:
    from collections.abc import Mapping

Solution = tuple[sp.Expr, ...]


def systeme(augmentee: sp.Matrix) -> SystemeLineaire:
    """Le systeme dont voici la matrice augmentee."""
    return SystemeLineaire(augmentee[:, :-1], augmentee[:, -1])


def nature(augmentee: sp.Matrix) -> Nature:
    """Impossible, unique ou indetermine."""
    return nature_des_solutions(systeme(augmentee))


def _avec_des_libres_imposees(
    augmentee: sp.Matrix, libres: tuple[int, ...]
) -> Solution:
    inconnues = sp.symbols(f"x1:{augmentee.cols}")
    equations = list(augmentee[:, :-1] * sp.Matrix(inconnues) - augmentee[:, -1])
    liees = [x for i, x in enumerate(inconnues) if i not in libres]
    (resolue,) = sp.solve(equations, liees, dict=True)
    noms = {inconnues[i]: PARAMETRES[rang] for rang, i in enumerate(libres)}
    return tuple(_affine(resolue.get(x, x).subs(noms)) for x in inconnues)


def _affine(expression: sp.Expr) -> sp.Expr:
    """Un point, plus chaque parametre fois sa direction : la forme du syllabus."""
    point = sp.factor(sp.cancel(expression.subs(dict.fromkeys(PARAMETRES, 0))))
    termes = [p * sp.cancel(sp.diff(expression, p)) for p in PARAMETRES]
    return sp.Add(point, *termes)


def solution(augmentee: sp.Matrix, libres: tuple[int, ...] = ()) -> Solution:
    """La solution generale ; `libres` impose les inconnues laissees libres."""
    if libres:
        return _avec_des_libres_imposees(augmentee, libres)
    generale = par_echelonnement_simple(systeme(augmentee)).forme_generale()
    return tuple(_affine(x) for x in generale)


def _exiger_le_cas(
    table: Mapping[int, Mapping[str, object]], numero: int, cle: str
) -> None:
    if cle not in table.get(numero, {}):
        msg = f"exercice {numero} : pas de cas {cle!r}"
        raise KeyError(msg)


def forme_du_cas(numero: int, cle: str) -> sp.Matrix:
    """La matrice augmentee du cas, echelonnee : les conditions s'y lisent."""
    _exiger_le_cas(enonces.FORMES, numero, cle)
    augmentee = enonces.PARAMETRIQUES[numero].subs(enonces.FORMES[numero][cle])
    inconnues = augmentee.cols - 1
    return echelonner(augmentee, colonnes=inconnues).matrice.applyfunc(sp.factor)


def solution_du_cas(numero: int, cle: str) -> Solution:
    """La solution generale d'un cas compatible."""
    _exiger_le_cas(enonces.SOLUTIONS, numero, cle)
    substitution, libres = enonces.SOLUTIONS[numero][cle]
    return solution(enonces.PARAMETRIQUES[numero].subs(substitution), libres)


def determinant(numero: int) -> sp.Expr:
    """Le determinant de la matrice du systeme, factorise."""
    return sp.factor(systeme(enonces.PARAMETRIQUES[numero]).matrice.det())


def generique(numero: int) -> Solution:
    """La solution unique, hors des racines du determinant."""
    generale = solution_generale(systeme(enonces.PARAMETRIQUES[numero]))
    return tuple(sp.factor(x) for x in generale)
