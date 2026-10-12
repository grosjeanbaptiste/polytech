"""Le produit vectoriel et le produit mixte, tels que les cours les ecrivent.

Sources : `MecaniqueRationnelle1/SYLLABUS/Syllabus_Partie_1_Statique.pdf`,
annexe A, pages 148 et 152 -- proprietes 1.9.3, relations (A.75) et (A.78) --
et le syllabus d'introduction aux sciences de l'ingenieur, chapitre N.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.produit_vectoriel import (
    matrice_antisymetrique,
    produit_mixte,
    produit_vectoriel,
)

A = sp.symbols("a_x a_y a_z")
B = sp.symbols("b_x b_y b_z")
C = sp.symbols("c_x c_y c_z")
UX, UY, UZ = (1, 0, 0), (0, 1, 0), (0, 0, 1)


def test_les_composantes_sont_celles_de_la_relation_a_75() -> None:
    """Page 152, (A.75) : `(ay bz - az by, az bx - ax bz, ax by - ay bx)`."""
    ax, ay, az = A
    bx, by, bz = B
    attendu = (ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx)
    assert produit_vectoriel(A, B) == attendu


def test_la_base_cartesienne_est_positivement_orientee() -> None:
    """Page 152 : `ux ^ uy = uz`, `uy ^ uz = ux`, `uz ^ ux = uy`."""
    assert produit_vectoriel(UX, UY) == UZ
    assert produit_vectoriel(UY, UZ) == UX
    assert produit_vectoriel(UZ, UX) == UY


def test_le_produit_vectoriel_est_anticommutatif_et_distributif() -> None:
    """Page 148, proprietes 2 et 4."""
    oppose = tuple(-composante for composante in produit_vectoriel(B, A))
    assert produit_vectoriel(A, B) == oppose
    somme = tuple(b + c for b, c in zip(B, C, strict=True))
    separes = zip(produit_vectoriel(A, B), produit_vectoriel(A, C), strict=True)
    assert all(
        sp.expand(ensemble - gauche - droite) == 0
        for ensemble, (gauche, droite) in zip(
            produit_vectoriel(A, somme), separes, strict=True
        )
    )


def test_le_produit_de_deux_vecteurs_paralleles_est_nul() -> None:
    """Page 148, caracteristiques : nul si les facteurs sont paralleles."""
    assert produit_vectoriel((1, 2, 3), (-2, -4, -6)) == (0, 0, 0)


def test_la_matrice_antisymetrique_rend_le_meme_produit() -> None:
    """Page 152, (A.78) : `a ^ b = [a~] b`."""
    par_la_matrice = matrice_antisymetrique(A) * sp.Matrix(B)
    assert tuple(par_la_matrice) == produit_vectoriel(A, B)


def test_le_produit_mixte_est_le_volume_du_parallelepipede() -> None:
    """Introduction aux sciences de l'ingenieur, page 136 : `a . (b ^ c)`."""
    assert produit_mixte((2, 0, 0), (0, 3, 0), (0, 0, 4)) == 24
    assert produit_mixte((0, 3, 0), (2, 0, 0), (0, 0, 4)) == -24
    assert produit_mixte((1, 2, 3), (2, 4, 6), (5, 7, 11)) == 0


def test_un_vecteur_qui_n_est_pas_de_l_espace_est_refuse() -> None:
    """Le produit vectoriel n'existe qu'en dimension trois."""
    with pytest.raises(ValueError, match="trois composantes attendues, et non"):
        produit_vectoriel((1, 2), (3, 4, 5))
