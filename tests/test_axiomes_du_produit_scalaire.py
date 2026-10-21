"""Syllabus d'algebre IX.1, page 135 : les trois proprietes d'un produit scalaire.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, page 135 (imprimee) :
1) `<X|X> >= 0` et `<X|X> = 0 <=> X = 0` ; 2) linearite en la premiere
place ; 3) symetrie. L'exercice 17 (page 143) demande de montrer que
`tr(A B^t)` est un produit scalaire sur les matrices carrees.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.axiomes import axiomes_du_produit_scalaire
from polytech.maths.domain.algebre.produits_scalaires import usuel


def _vecteur(nom: str) -> sp.Matrix:
    return sp.Matrix(sp.symbols(f"{nom}1:4", real=True))


def _matrice(nom: str) -> sp.Matrix:
    return sp.Matrix(2, 2, sp.symbols(f"{nom}1:5", real=True))


def _trace(a: sp.Matrix, b: sp.Matrix) -> sp.Expr:
    return (a * b.T).trace()


def test_le_produit_usuel_verifie_les_trois_proprietes() -> None:
    assert axiomes_du_produit_scalaire(usuel, _vecteur) == {
        "positivite": True,
        "linearite": True,
        "symetrie": True,
    }


def test_la_trace_de_a_b_transposee_est_un_produit_scalaire() -> None:
    """Exercice 17, page 143."""
    assert all(axiomes_du_produit_scalaire(_trace, _matrice).values())


def test_une_forme_symetrique_non_positive_n_est_pas_un_produit_scalaire() -> None:
    def croise(x: sp.Matrix, y: sp.Matrix) -> sp.Expr:
        return x[0] * y[1] + x[1] * y[0] + x[2] * y[2]

    verdict = axiomes_du_produit_scalaire(croise, _vecteur)
    assert verdict == {"positivite": False, "linearite": True, "symetrie": True}


def test_une_forme_positive_mais_degeneree_echoue_a_la_positivite() -> None:
    """`<X|X> = 0` pour `X = (0, 0, 1)` non nul."""

    def partiel(x: sp.Matrix, y: sp.Matrix) -> sp.Expr:
        return x[0] * y[0] + x[1] * y[1]

    assert not axiomes_du_produit_scalaire(partiel, _vecteur)["positivite"]


def test_une_forme_non_symetrique_le_dit() -> None:
    def penche(x: sp.Matrix, y: sp.Matrix) -> sp.Expr:
        return x[0] * y[0] + x[0] * y[1] + x[1] * y[1] + x[2] * y[2]

    assert not axiomes_du_produit_scalaire(penche, _vecteur)["symetrie"]


def test_une_forme_non_lineaire_le_dit() -> None:
    def carre(x: sp.Matrix, y: sp.Matrix) -> sp.Expr:
        return x[0] ** 2 * y[0] ** 2 + x[1] * y[1] + x[2] * y[2]

    assert not axiomes_du_produit_scalaire(carre, _vecteur)["linearite"]


def test_un_produit_pondere_positif_est_accepte() -> None:
    def pondere(x: sp.Matrix, y: sp.Matrix) -> sp.Expr:
        return 2 * x[0] * y[0] + 3 * x[1] * y[1] + x[2] * y[2]

    assert all(axiomes_du_produit_scalaire(pondere, _vecteur).values())


def test_une_forme_affine_n_est_pas_positive() -> None:
    """`<X|X> = 1 + ...` ne s'annule jamais : `X = 0` y echoue."""

    def decale(x: sp.Matrix, y: sp.Matrix) -> sp.Expr:
        return 1 + x[0] * y[0] + x[1] * y[1] + x[2] * y[2]

    assert not axiomes_du_produit_scalaire(decale, _vecteur)["positivite"]


def test_un_carre_non_polynomial_est_hors_de_portee() -> None:
    def exponentiel(x: sp.Matrix, y: sp.Matrix) -> sp.Expr:
        return sp.exp(x[0] * y[0]) - 1 + x[1] * y[1] + x[2] * y[2]

    with pytest.raises(ValueError, match="hors de portee"):
        axiomes_du_produit_scalaire(exponentiel, _vecteur)
