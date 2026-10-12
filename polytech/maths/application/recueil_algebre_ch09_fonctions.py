"""Recueil d'algebre, chapitre IX, exercices 18 a 33 : matrices, polynomes, fonctions.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 143 a 150 :
le produit scalaire de la trace, les produits scalaires integraux et aux
points, les moindres carres, et deux sujets d'examen.
"""

from __future__ import annotations

from typing import Any, Final

import sympy as sp

from polytech.maths.domain.algebre import moindres_carres as mc
from polytech.maths.domain.algebre import orthogonalisation as orth
from polytech.maths.domain.algebre import produits_scalaires as ps
from polytech.maths.domain.algebre.rang_exact import rangs_selon_le_parametre

M = sp.Matrix
X: Final = sp.Symbol("x")
PARAMETRE: Final = sp.Symbol("a")
TRACE = ps.de_la_trace
UN: Final = sp.Integer(1)
MONOMES: Final = (UN, X, X**2)
LEGENDRE: Final = ps.integral(X, -1, 1)
SUR_LA_PERIODE: Final = ps.integral(X, 0, 2 * sp.pi)
CANONIQUE: Final = tuple(M(2, 2, [int(i == k) for i in range(4)]) for k in range(4))
B_18: Final = (M([[1, 0], [0, 0]]), M([[0, 1], [1, 1]]), M([[1, 1], [1, 2]]))
B_19: Final = (M([[1, 0], [0, 0]]), M([[0, 1], [1, 1]]), M([[0, 0], [0, 1]]))
W_20: Final = (M([[1, 0], [2, 1]]), M([[2, 1], [2, 3]]), M([[0, 1], [-2, 1]]))
M_18: Final = M([[1, 2], [3, 4]])
A_20: Final = M([[1, 0], [3, 1]])
F_26: Final = (UN, sp.sin(X), sp.cos(X), sp.sin(5 * X))
W_29: Final = (UN, sp.cos(X), sp.cos(2 * X), sp.sin(X), sp.sin(2 * X))
SYSTEME_30: Final = (M([[0, 0, 1], [0, 1, 1], [1, 1, 1], [0, 1, 0]]), M([1, 0, 1, 1]))
POINTS_31: Final = ((0, 1), (1, 3), (2, 2), (3, 4), (4, 5))
P_32: Final = 10 * X**3 - 3 * X**2 + 1
W_33: Final = (M([[0, 1], [-1, 0]]), M([[1, 1], [1, -1]]), M([[-1, 1], [PARAMETRE, 1]]))
A_33: Final = M([[1, 4], [1, -1]])


def _gs(famille: tuple[Any, ...], produit: ps.Produit) -> tuple[Any, ...]:
    return orth.gram_schmidt(famille, produit)


def _matrices() -> dict[str, Any]:
    orthogonal = orth.orthogonal(W_20[:2], CANONIQUE, TRACE)
    sur_orthogonal = orth.projection(A_20, _gs(orthogonal, TRACE), TRACE)
    base_18 = _gs(B_18, TRACE)
    return {
        "18a": orth.normaliser(base_18, TRACE),
        "18b": orth.projection(M_18, base_18, TRACE),
        "19": orth.distance_au_sous_espace(M_18, _gs(B_19, TRACE), TRACE),
        "20Orthogonal": orthogonal,
        "20SurOrthogonal": sur_orthogonal,
        "20SurW": A_20 - sur_orthogonal,
    }


def _polynomes_et_fonctions() -> dict[str, Any]:
    legendre = _gs(MONOMES, LEGENDRE)
    droites = _gs(MONOMES[:2], LEGENDRE)
    quatre_points = ps.aux_points(X, (-1, 0, 1, 2))
    plus_proche = orth.projection(sp.exp(X), droites, LEGENDRE).rewrite(sp.exp)
    ecart = sp.exp(X) - plus_proche
    return {
        "21Produit": LEGENDRE(UN, X**2),
        "21Norme": ps.norme(X, LEGENDRE),
        "22": legendre,
        "23": _gs(MONOMES, ps.aux_points(X, (-2, -1, 0, 1, 2))),
        "24": orth.orthogonal((UN, X**2), MONOMES, quatre_points),
        "25": legendre,
        "26": orth.normaliser(F_26, ps.integral(X, -sp.pi, sp.pi)),
        "27a": plus_proche,
        "27b": sp.sqrt(sp.simplify(LEGENDRE(ecart, ecart).rewrite(sp.exp))),
        "28": orth.projection(X, W_29[:1] + W_29[1:4:2], SUR_LA_PERIODE),
        "29": orth.projection(1 + X, W_29, SUR_LA_PERIODE),
        "32": orth.distance_au_sous_espace(P_32, legendre, LEGENDRE),
    }


def reponses() -> dict[str, Any]:
    """Chaque reponse des exercices 18 a 33, par sa cle."""
    singulier = tuple(m.subs(PARAMETRE, -3) for m in W_33[:2])
    projete = orth.projection(A_33, _gs(singulier, TRACE), TRACE)
    examens = {
        "30": mc.au_sens_des_moindres_carres(*SYSTEME_30),
        "31": mc.droite_de_regression(POINTS_31),
        "33Generique": sp.simplify(orth.projection(A_33, _gs(W_33, TRACE), TRACE)),
        "33Projection": projete,
        "33SurOrthogonal": A_33 - projete,
    }
    return _matrices() | _polynomes_et_fonctions() | examens


def exercice_26_est_orthogonale() -> bool:
    """Vrai si `{1, sin x, cos x, sin 5x}` est orthogonale sur `[-pi, pi]`."""
    return ps.est_orthogonale(F_26, ps.integral(X, -sp.pi, sp.pi))


def dimension_du_33() -> tuple[int, dict[sp.Expr, int]]:
    """La dimension de `W` hors des valeurs singulieres de `a`, et en chacune."""
    lignes = M([list(m) for m in W_33])
    return rangs_selon_le_parametre(lignes, PARAMETRE)


def extremites_du_31() -> tuple[tuple[sp.Expr, sp.Expr], tuple[sp.Expr, sp.Expr]]:
    """Les deux points de la droite de regression aux abscisses extremes."""
    pente, ordonnee = mc.droite_de_regression(POINTS_31)
    debut, fin = sp.Integer(POINTS_31[0][0]), sp.Integer(POINTS_31[-1][0])
    return (debut, pente * debut + ordonnee), (fin, pente * fin + ordonnee)
