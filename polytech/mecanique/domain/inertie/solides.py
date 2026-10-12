"""Les solides homogenes des exercices du cours, comme repartitions de masse.

« En general, le systeme sera homogene. La masse elementaire `dm` est
alors remplacee par `rho dV`. » Chaque solide est decrit dans le repere
ou le cours le pose : la couronne et le cylindre autour de l'axe `z`, la
tige le long de `x` depuis son extremite, la brique par ses trois cotes
et le coin d'ou elle part -- centree sur l'origine par defaut.

Source : `MecaniqueRationnelle2/SYLLABUS/Syllabus_Theorie_2025_2026.pdf`,
section 6.7, pages 114 et 115, exercices 6.7.1 a 6.7.5 ; la tige est celle
de la figure 6.1, page 102.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.mecanique.domain.inertie.repartition import Repartition

if TYPE_CHECKING:
    from polytech.maths.domain.algebre.produit_vectoriel import Vecteur

X, Y, Z = sp.symbols("x y z", real=True)
RAYON, ANGLE = sp.symbols("r theta", nonnegative=True)
ZERO = sp.Integer(0)


def tige(masse: sp.Expr, longueur: sp.Expr) -> Repartition:
    """Une tige de `O` a `x = L` : `dm = (m / L) dx`."""
    return Repartition((X, ZERO, ZERO), masse / longueur, ((X, ZERO, longueur),))


def brique(
    masse: sp.Expr, cotes: Vecteur, origine: Vecteur | None = None
) -> Repartition:
    """Un rectangle plein de cotes `(a, b, c)` ; `origine` est son coin minimal."""
    a, b, c = cotes
    x0, y0, z0 = origine if origine is not None else (-a / 2, -b / 2, -c / 2)
    bornes = ((X, x0, x0 + a), (Y, y0, y0 + b), (Z, z0, z0 + c))
    return Repartition((X, Y, Z), masse / (a * b * c), bornes)


def couronne(masse: sp.Expr, exterieur: sp.Expr, interieur: sp.Expr) -> Repartition:
    """Une couronne plane d'axe `z` ; `interieur = 0` donne le cercle plein."""
    hors_bornes = sp.Or(sp.Lt(interieur, 0), sp.Ge(interieur, exterieur))
    if hors_bornes is sp.true:
        msg = f"rayon interieur {interieur} hors de [0, {exterieur}["
        raise ValueError(msg)
    aire = sp.pi * (exterieur**2 - interieur**2)
    position = (RAYON * sp.cos(ANGLE), RAYON * sp.sin(ANGLE), ZERO)
    bornes = ((RAYON, interieur, exterieur), (ANGLE, ZERO, 2 * sp.pi))
    return Repartition(position, masse / aire * RAYON, bornes)


def cylindre(masse: sp.Expr, rayon: sp.Expr, hauteur: sp.Expr) -> Repartition:
    """Un cylindre droit plein d'axe `z`, centre sur l'origine."""
    volume = sp.pi * rayon**2 * hauteur
    position = (RAYON * sp.cos(ANGLE), RAYON * sp.sin(ANGLE), Z)
    bornes = (
        (RAYON, ZERO, rayon),
        (ANGLE, ZERO, 2 * sp.pi),
        (Z, -hauteur / 2, hauteur / 2),
    )
    return Repartition(position, masse / volume * RAYON, bornes)
