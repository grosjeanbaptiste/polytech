"""Mathematiques I, chapitre 3 d'algebre : les espaces vectoriels.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Ces tests
verifient que les symboles auxquels `cursus/notions/AlgebreLineaire1.toml`
rattache un titre portent ce que la planche enonce -- en recopiant ses
exemples, page citee.

Source : `AlgebreLineaire1/SLIDES/Slides_03_Espaces_vectoriels.pdf`
(« ch. 3 »), et la page 113 du jeu du chapitre 2. Les pages sont celles du
PDF.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.domain.algebre import familles_de_vecteurs as familles
from polytech.maths.domain.algebre import sous_espaces
from polytech.maths.domain.algebre.echelonnement import (
    echelonner,
    echelonner_simplement,
    est_echelonnee,
    est_echelonnee_simple,
)
from polytech.maths.domain.algebre.espaces import combinaison_lineaire
from polytech.maths.domain.algebre.matrices_particulieres import (
    parties_symetrique_et_antisymetrique,
)
from polytech.maths.domain.algebre.rang import est_inversible
from polytech.maths.domain.algebre.rang_exact import rang_par_echelonnement

TROIS = ((1, 1, 1), (4, 2, 3), (3, 4, 3))
"""Ch. 3, pages 12 a 16 : la famille libre et generatrice de l'exemple."""

CINQ = sp.Matrix(
    [
        [1, -2, 3, -1, 2],
        [2, -1, 2, 2, 4],
        [-3, -5, 1, -5, -4],
        [3, 1, 2, 3, 5],
        [6, 2, 4, 6, 10],
    ]
)
"""Ch. 3, page 76 : la matrice du calcul de rang."""


def test_la_combinaison_lineaire_de_la_page_5() -> None:
    """Ch. 3, page 5 ; ch. 2, page 113 : `(7, 14, 10)`, puis `(3, 1, 7)`."""
    famille = [sp.Matrix(v) for v in ((1, 2, 3), (1, 0, -1), (2, 3, 1), (0, 1, 0))]
    assert combinaison_lineaire((2, -1, 3, 1), famille) == sp.Matrix([7, 14, 10])
    deux = [sp.Matrix([1, 2, 3]), sp.Matrix([2, -1, 4])]
    assert combinaison_lineaire((1, 1), deux) == sp.Matrix([3, 1, 7])


def test_la_famille_des_pages_12_a_16_est_une_base() -> None:
    """Ch. 3, pages 12 et 16 : base, `(3, 2, 1) = -14 X1 + 2 X2 + 3 X3`."""
    assert familles.est_libre(TROIS)
    assert familles.est_generatrice(TROIS)
    assert familles.est_une_base(TROIS)
    assert familles.coordonnees((3, 2, 1), TROIS) == (-14, 2, 3)


def test_les_deux_familles_liees_des_pages_20_et_22() -> None:
    """Ch. 3, pages 20 et 22 : liee et generatrice, puis liee et non generatrice."""
    quatre = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1))
    assert not familles.est_libre(quatre)
    assert familles.est_generatrice(quatre)
    a, b, c, d = familles.relation_de_dependance(quatre)
    assert (a / d, b / d, c / d) == (-1, -1, -1)
    deux = ((1, 1, 0), (2, 2, 0))
    assert not familles.est_libre(deux)
    assert not familles.est_generatrice(deux)
    assert not familles.appartient((1, 2, 3), deux)


def test_le_prolongement_de_la_page_51() -> None:
    """Ch. 3, page 51 : `{(1, 1, 0)}` devient `{(1, 1, 0), (1, 0, 0), (0, 0, 1)}`."""
    assert familles.prolonger_en_base(((1, 1, 0),)) == (
        (1, 1, 0),
        (1, 0, 0),
        (0, 0, 1),
    )
    generatrice = ((1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 1))
    assert len(familles.extraire_une_base(generatrice)) == len(generatrice[0])


def test_le_rang_et_les_formes_echelonnees_de_la_page_76() -> None:
    """Ch. 3, page 76, rendue en image : `rg(A) = 3`, et la forme simple."""
    assert rang_par_echelonnement(CINQ) == 3
    assert not est_inversible(
        tuple(tuple(float(c) for c in ligne) for ligne in CINQ.tolist())
    )
    echelonnee = echelonner(CINQ).matrice
    assert est_echelonnee(echelonnee)
    assert 3 * echelonnee.row(2) == sp.Matrix([[0, 0, -14, 20, 6]])
    simple = echelonner_simplement(CINQ)
    assert est_echelonnee_simple(simple)
    septiemes = [[15, 15], [-4, -4], [-10, -3]]
    assert 7 * simple[:3, 3:] == sp.Matrix(septiemes)


def test_les_sommes_de_sous_espaces_des_pages_71_et_87() -> None:
    """Ch. 3, pages 71, 87 et 92 : droite et plan ; parties d'une matrice."""
    droite, plan = ((1, 2, 3),), ((1, 1, 0), (0, 1, 1))
    assert sous_espaces.sont_supplementaires(droite, plan)
    assert sous_espaces.intersection(droite, plan) == ()
    assert len(sous_espaces.somme(droite, plan)) == len(droite) + len(plan)
    symetrique, antisymetrique = parties_symetrique_et_antisymetrique(
        sp.Matrix([[3, 4], [6, 7]])
    )
    assert symetrique == sp.Matrix([[3, 5], [5, 7]])
    assert antisymetrique == sp.Matrix([[0, -1], [1, 0]])
