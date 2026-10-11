"""Mathematiques I, chapitre 4 d'algebre : les applications lineaires.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Ces tests
verifient que les symboles auxquels `cursus/notions/AlgebreLineaire1.toml`
rattache un titre portent ce que la planche enonce -- en recopiant ses
exemples, page citee.

Source : `AlgebreLineaire1/SLIDES/Slides_04_Applications_lineaires.pdf`.
Les pages sont celles du PDF ; la planche imprimee est entre parentheses.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.algebre import representation
from polytech.maths.domain.algebre.applications_lineaires import ApplicationLineaire
from polytech.maths.domain.algebre.familles_de_vecteurs import coordonnees
from polytech.maths.domain.algebre.linearite import (
    existe_une_application_lineaire,
    respecte_les_combinaisons,
)
from polytech.maths.domain.algebre.sous_espaces import equations

V = ((1, 0, 0), (1, 1, 0), (1, 1, 1))
V_PRIME = ((0, 1, 1), (1, 0, 1), (1, 1, 0))
W = ((1, 0), (1, 1))
W_PRIME = ((1, 1), (1, -1))
"""Pages 48 a 81 (16 a 28) : les quatre bases de l'exemple suivi."""

F = ApplicationLineaire.depuis_la_regle(
    lambda x, y, z: (2 * x - y + 3 * z, -x + 2 * y + z), 3
)
"""Page 52 (17) : `f(x, y, z) = (2x - y + 3z, -x + 2y + z)`."""

ASSOCIEE = sp.Matrix([[3, 0, 2], [-1, 1, 2]])
"""Page 52 (17) : `A_f(v, w)`."""


def test_l_application_de_la_page_18_est_injective_et_son_image_est_un_plan() -> None:
    """Page 18 (6) : `f(a, b) = (a, a + b, b)`, `Ker f = {0}`, `Im f : y = x + z`."""
    f = ApplicationLineaire.depuis_la_regle(lambda a, b: (a, a + b, b), 2)
    assert f.est_injective()
    assert not f.est_surjective()
    assert f.rang() == 2
    ((a, b, c),) = equations(f.image())
    assert (a / a, b / a, c / a) == (1, -1, 1)
    assert respecte_les_combinaisons(
        lambda v: sp.Matrix([v[0], v[0] + v[1], v[1]]),
        sp.Matrix(sp.symbols("a b")),
        sp.Matrix(sp.symbols("c d")),
    )


def test_la_transformation_de_la_page_22_a_une_droite_pour_noyau() -> None:
    """Page 22 (7) : `f(a, b, c) = (a + c, b + c, a - b)`, noyau `(1, 1, -1)`."""
    f = ApplicationLineaire.depuis_la_regle(lambda a, b, c: (a + c, b + c, a - b), 3)
    (direction,) = f.noyau()
    assert sp.Matrix(direction).cross(sp.Matrix([1, 1, -1])).is_zero_matrix
    assert f.rang() + len(f.noyau()) == 3
    ((a, b, c),) = equations(f.image())
    assert (a / a, b / a, c / a) == (1, -1, -1)


def test_les_images_d_une_base_determinent_l_application() -> None:
    """Page 34 (11), propriete 4 : une seule `f` telle que `f(v_i) = Y_i`."""
    images = ((2, -1), (1, 1), (4, 2))
    assert existe_une_application_lineaire(V, images)
    f = ApplicationLineaire.depuis_les_images(V, images)
    assert f.matrice == F.matrice
    assert not existe_une_application_lineaire(((1, 0), (2, 0)), ((1,), (3,)))


def test_les_coordonnees_et_la_matrice_associee_des_pages_48_a_54() -> None:
    """Pages 48, 52 et 54 (16, 17 et 19) : `X_v`, `A_f(v, w)`, `Y_w = A X_v`."""
    assert coordonnees((4, 3, 1), V) == (1, 2, 1)
    assert representation.matrice_associee(F.matrice, V, W) == ASSOCIEE
    assert representation.matrice_canonique(ASSOCIEE, V, W) == F.matrice
    assert F.image_de((4, 3, 1)) == (8, 3)
    assert ASSOCIEE * sp.Matrix([1, 2, 1]) == sp.Matrix(coordonnees((8, 3), W))
    assert coordonnees((8, 3), W) == (5, 3)


def test_les_matrices_de_passage_des_pages_65_et_70() -> None:
    """Pages 59, 65 et 70 (22, 24 et 25) : `X_v' = P(v, v') X_v`, `P P^-1 = I`."""
    passage = representation.matrice_de_passage(V, V_PRIME)
    assert passage == sp.Matrix([[-1, 0, 1], [1, 0, 1], [1, 2, 1]]) / 2
    assert passage * sp.Matrix([1, 2, 1]) == sp.Matrix([0, 1, 3])
    assert coordonnees((4, 3, 1), V_PRIME) == (0, 1, 3)
    de_w = representation.matrice_de_passage(W, W_PRIME)
    assert de_w == sp.Matrix([[1, 2], [1, 0]]) / 2
    assert representation.matrice_de_passage(W_PRIME, W) == de_w.inv()


def test_le_changement_de_bases_de_la_page_81() -> None:
    """Page 81 (28) : `A(v', w') = P(w, w') A(v, w) P(v, v')^-1`."""
    attendue = sp.Matrix([[5, 5, 2], [-1, 5, 0]]) / 2
    par_les_passages = (
        representation.matrice_de_passage(W, W_PRIME)
        * ASSOCIEE
        * representation.matrice_de_passage(V, V_PRIME).inv()
    )
    assert par_les_passages == attendue
    assert representation.matrice_associee(F.matrice, V_PRIME, W_PRIME) == attendue
