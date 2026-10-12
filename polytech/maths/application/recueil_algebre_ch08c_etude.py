"""Recueil d'algebre, chapitre VIII, section VIII.8 : noyau, image, discussion.

Chaque sujet demande la matrice, une base du noyau, une base de l'image, et
le plus souvent une discussion. `noyau` et `image` les rendent pour le cas
general (cle vide) ou pour un cas singulier de `enonces.CAS`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.maths.application import recueil_algebre_ch08c_enonces as enonces
from polytech.maths.domain.algebre.applications_lineaires import ApplicationLineaire
from polytech.maths.domain.algebre.solutions import PARAMETRES

if TYPE_CHECKING:
    from polytech.maths.application.recueil_algebre_ch07_familles import Base

GENERAL = ""
"""La cle du cas general : aucune valeur n'est donnee aux parametres."""


def matrice(numero: int, cle: str = GENERAL) -> sp.Matrix:
    """La matrice de l'exercice, dans le cas demande ; leve s'il n'existe pas."""
    if numero not in enonces.APPLICATIONS:
        msg = f"pas d'exercice {numero} dans la section VIII.8"
        raise KeyError(msg)
    generale = enonces.APPLICATIONS[numero]
    if cle == GENERAL:
        return generale
    if cle not in enonces.CAS.get(numero, {}):
        msg = f"exercice {numero} : pas de cas {cle!r}"
        raise KeyError(msg)
    return generale.subs(enonces.CAS[numero][cle])


def noyau(numero: int, cle: str = GENERAL) -> Base:
    """Une base de `Ker f`."""
    base = ApplicationLineaire(matrice(numero, cle)).noyau()
    return tuple(tuple(sp.simplify(x) for x in vecteur) for vecteur in base)


def image(numero: int, cle: str = GENERAL) -> Base:
    """Une base de `Im f`, extraite des colonnes de la matrice."""
    return ApplicationLineaire(matrice(numero, cle)).image()


def rang(numero: int, cle: str = GENERAL) -> int:
    """La dimension de l'image."""
    return len(image(numero, cle))


def determinant(numero: int) -> sp.Expr:
    """Le determinant factorise d'une matrice carree : ses racines sont les cas."""
    return sp.factor(matrice(numero).det())


def antecedents(numero: int, cible: tuple[object, ...]) -> sp.Set:
    """Tous les antecedents de `cible`, parametres par `alpha`, `beta` ; ou le vide."""
    solutions = sp.linsolve((matrice(numero), sp.Matrix(list(cible))))
    libres = sorted(solutions.free_symbols - matrice(numero).free_symbols, key=str)
    libres = [s for s in libres if s.is_Dummy or str(s).startswith("tau")]
    return solutions.subs(dict(zip(libres, PARAMETRES, strict=False)))


def inverse_du_36() -> sp.Matrix:
    """La matrice de `f^-1` quand `a != +-b`."""
    return sp.simplify(matrice(36).inv())
