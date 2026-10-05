"""Syllabus d'algebre, chapitre VII, sections 1 a 6 : sous-espaces de `K^n`.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 71 a 86, et
les solutions des pages 94 et 95. Les exemples 1 a 9 -- sous-espace,
famille liee par trois methodes, famille generatrice, base extraite, base
prolongee, coordonnees, equations d'un sous-espace -- puis les reponses
officielles des exercices 1 a 22.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import sympy as sp

from polytech.maths.domain.algebre.determinants import Determinant
from polytech.maths.domain.algebre.echelonnement import echelonner
from polytech.maths.domain.algebre.espaces import combinaison_lineaire
from polytech.maths.domain.algebre.familles_de_vecteurs import (
    appartient,
    coordonnees,
    est_generatrice,
    est_libre,
    est_une_base,
    extraire_une_base,
    prolonger_en_base,
    rang_de_la_famille,
    relation_de_dependance,
)
from polytech.maths.domain.algebre.sous_espaces import (
    base_du_noyau,
    engendrent_le_meme_sous_espace,
    equations,
    intersection,
    somme,
    sont_supplementaires,
)
from polytech.maths.domain.algebre.stabilite import (
    contre_exemple,
    est_lineaire_homogene,
)

if TYPE_CHECKING:
    from collections.abc import Callable

M = sp.Matrix
R = sp.Rational
I = sp.I  # noqa: E741 -- l'unite imaginaire
X1, X2, X3 = sp.symbols("x1 x2 x3", real=True)
F1 = ((1, 2, -1), (2, 4, 1), (3, 6, 3))
F2 = ((1, 2, 3), (2, 3, 4))
F3 = ((1, 2, 3), (4, 3, 5), (1, 1, 1), (3, 1, 2))
CINQ = ((1, 1, 2, 1), (1, -1, 0, 1), (1, 2, 2, 0), (1, 1, 1, 0), (-2, 5, 4, -1))
W1 = ((1, -1, 2, 1), (2, 0, -1, 0), (1, 3, -8, -3))
W2 = ((1, 5, 2, -1), (0, -3, 0, 1))
GRILLE = tuple(M([a, b, c]) for a in (-1, 0, 1) for b in (-1, 0, 1) for c in (-1, 0, 1))


def _dans_r3(condition: sp.Basic) -> Callable[[sp.Matrix], bool]:
    """L'appartenance a `{(x1, x2, x3) | condition}`."""
    return lambda v: bool(condition.subs({X1: v[0], X2: v[1], X3: v[2]}))


def test_l_exemple_1_est_un_sous_espace() -> None:
    """`W = {(x, y, z) | x - 2y + 3z = 0}` : une equation lineaire homogene."""
    equation = X1 - 2 * X2 + 3 * X3
    assert est_lineaire_homogene([equation], (X1, X2, X3))
    assert contre_exemple(_dans_r3(sp.Eq(equation, 0)), GRILLE) is None


def test_l_exemple_2_par_les_combinaisons_lineaires() -> None:
    """`(3, 6, 3) = -1 (1, 2, -1) + 2 (2, 4, 1)` : la famille `F1` est liee."""
    assert coordonnees(F1[2], F1[:2]) == (-1, 2)
    assert not est_libre(F1)


def test_l_exemple_2_par_le_rang() -> None:
    """`rg = 2 < 3`, et les operations redonnent `X3 = -X1 + 2 X2`."""
    resultat = echelonner(M(F1))
    assert resultat.matrice == M([[1, 2, -1], [0, 0, 3], [0, 0, 0]])
    assert resultat.operations == (
        "l2 -> l2 - 2 l1",
        "l3 -> l3 - 3 l1",
        "l3 -> l3 - 2 l2",
    )
    assert rang_de_la_famille(F1) == 2
    assert M(F1[2]) == -M(F1[0]) + 2 * M(F1[1])


def test_l_exemple_2_par_la_definition_de_famille_libre() -> None:
    """`gamma = 1`, `alpha = 1`, `beta = -2` : `X1 - 2 X2 + X3 = 0`."""
    assert relation_de_dependance(F1) == (1, -2, 1)


def test_une_famille_libre_n_a_pas_de_relation() -> None:
    with pytest.raises(ValueError, match="famille libre"):
        relation_de_dependance(F2)


def test_l_exemple_3_donne_une_base_du_plan() -> None:
    """`alpha = 1` puis `beta = 1` : `{(2, 1, 0), (-3, 0, 1)}`."""
    base = base_du_noyau(M([[1, -2, 3]]))
    assert base == ((2, 1, 0), (-3, 0, 1))
    assert est_libre(base)


def test_l_exemple_4_une_famille_liee_de_longueur_trois_n_engendre_pas_r3() -> None:
    assert not est_generatrice(F1)
    assert not est_une_base(F1)


def test_l_exemple_5_par_le_rang() -> None:
    """Rang 3 pour 5 vecteurs de `R^4` : ni libre, ni generatrice ; base extraite."""
    assert rang_de_la_famille(CINQ) == 3
    assert not est_libre(CINQ)
    assert not est_generatrice(CINQ)
    assert extraire_une_base(CINQ) == CINQ[:3]
    for vecteur in CINQ[3:]:
        assert appartient(vecteur, CINQ[:3])


def test_l_exemple_6_prolonge_une_famille_libre() -> None:
    """`F2` est libre, n'engendre pas `R^3`, et `(0, 0, 1)` la complete."""
    assert est_libre(F2)
    assert rang_de_la_famille(F2) == 2
    assert not est_generatrice(F2)
    assert not appartient((0, 0, 1), F2)
    assert est_une_base((*F2, (0, 0, 1)))
    prolongee = prolonger_en_base(F2)
    assert prolongee[:2] == F2
    assert est_une_base(prolongee)


def test_une_famille_liee_ne_se_prolonge_pas_en_base() -> None:
    with pytest.raises(ValueError, match="ne se prolonge pas en une base"):
        prolonger_en_base(F1)


def test_un_contre_exemple_se_cherche_parmi_des_candidats() -> None:
    with pytest.raises(ValueError, match="aucun candidat"):
        contre_exemple(lambda _: True, ())


def test_l_exemple_7_extrait_une_base() -> None:
    """`X10 = X8 - X7`, et `{X7, X8, X9}` est une base de `R^3`."""
    assert not est_libre(F3)
    assert coordonnees(F3[3], F3[:2]) == (-1, 1)
    assert extraire_une_base(F3) == F3[:3]
    assert rang_de_la_famille(F3[:3]) == 3
    assert est_une_base(F3[:3])
    assert est_generatrice(F3)


def test_l_exemple_8_donne_les_coordonnees() -> None:
    """`(4, 5, 6) = 1 X7 + 0 X8 + 3 X9`."""
    assert coordonnees((4, 5, 6), F3[:3]) == (1, 0, 3)


def test_des_coordonnees_demandent_une_famille_libre_et_un_vecteur_engendre() -> None:
    with pytest.raises(ValueError, match="famille liee"):
        coordonnees((4, 5, 6), F1)
    with pytest.raises(ValueError, match="n'est pas combinaison lineaire"):
        coordonnees((0, 0, 1), F2)


def test_l_exemple_9_donne_les_equations_du_sous_espace() -> None:
    """`-x + y + 2z = 0` et `3x + y + 2t = 0` : `4 - 2 = 2` equations."""
    famille = ((1, 1, 0, -2), (0, -2, 1, 1), (1, -1, 1, -1))
    assert extraire_une_base(famille) == famille[:2]
    obtenues = equations(famille)
    assert len(obtenues) == 2
    officielles = ((-1, 1, 2, 0), (3, 1, 0, 2))
    assert engendrent_le_meme_sous_espace(obtenues, officielles)
    for equation in officielles:
        for vecteur in famille:
            assert M(equation).dot(M(vecteur)) == 0


def test_le_lien_avec_les_systemes_lineaires() -> None:
    """VII.5 : `A X = x1 A1 + ... + xn An`, soluble si `b` est combili des colonnes."""
    a = M([[1, 1, 2], [2, 4, -3], [3, 6, -5]])
    colonnes = tuple(tuple(a.col(j)) for j in range(3))
    inconnues = sp.symbols("x y z")
    produit = a * M(inconnues)
    assert produit == combinaison_lineaire(inconnues, [a.col(j) for j in range(3)])
    assert est_libre(colonnes)
    assert coordonnees((9, 1, 0), colonnes) == (1, 2, 3)
    impossible = M([[1, 1, 0], [2, 3, 1], [-1, 3, 4]])
    colonnes = tuple(tuple(impossible.col(j)) for j in range(3))
    assert not est_libre(colonnes)
    assert not appartient((3, 9, 2), colonnes)


def test_la_reponse_de_l_exercice_1() -> None:
    """A, E, F, G : oui ; B, C, D : non."""
    variables = (X1, X2, X3)
    assert est_lineaire_homogene([X1 - X2], variables)
    assert est_lineaire_homogene([X1 + X2 + X3], variables)
    assert est_lineaire_homogene([X1 - X2, 2 * X2 - X3], variables)
    assert est_lineaire_homogene([X1 + X2 - 3 * X3], variables)
    b = contre_exemple(_dans_r3(X1 < X2), GRILLE)
    assert b is not None
    assert b.nature == "vecteur nul"
    c = contre_exemple(_dans_r3(sp.Eq(X1**2 + X2**2 + X3**2, 1)), GRILLE)
    assert c is not None
    assert c.nature == "vecteur nul"
    d = contre_exemple(_dans_r3(sp.Eq(X1 * X2, 0)), GRILLE)
    assert d is not None
    assert d.nature == "somme"
    premier, second = d.temoins
    assert premier[0] * premier[1] == 0
    assert second[0] * second[1] == 0
    assert (premier + second)[0] * (premier + second)[1] != 0
    assert not est_lineaire_homogene([X1 * X2], variables)


def test_la_reponse_de_l_exercice_2() -> None:
    """Un systeme homogene definit un sous-espace ; avec un second membre 3, non."""
    variables = (X1, X2, X3)
    assert est_lineaire_homogene([X1 + X2 + X3, 2 * X1 - X2], variables)
    assert not est_lineaire_homogene([X1 + X2 + X3, 2 * X1 - X2 - 3], variables)
    condition = sp.And(sp.Eq(X1 + X2 + X3, 0), sp.Eq(2 * X1 - X2, 3))
    violation = contre_exemple(_dans_r3(condition), GRILLE)
    assert violation is not None
    assert violation.nature == "vecteur nul"


def test_une_partie_instable_pour_le_produit_par_un_scalaire() -> None:
    """Le demi-espace `x1 >= 0` contient zero et les sommes, pas les opposes."""
    violation = contre_exemple(_dans_r3(X1 >= 0), GRILLE)
    assert violation is not None
    assert violation.nature == "multiple"
    scalaire, vecteur = violation.temoins
    assert vecteur[0] >= 0
    assert (scalaire * vecteur)[0] < 0


def test_les_reponses_des_exercices_3_a_5() -> None:
    """3 a) et b) : libres ; 4 : liee ; 5 : libres si et seulement si `a != 1`."""
    assert est_libre(((1, 1, 0), (1, 1, 1), (0, 1, -1)))
    assert est_libre(((0, 1, 1), (0, 2, 1), (1, 5, 3)))
    assert not est_libre(W1)
    a = sp.Symbol("a")
    famille = M([[1, 1, 0], [1, 1, 1], [a, 1, -1]])
    assert Determinant(famille).valeurs_singulieres(a) == (1,)


def test_les_reponses_des_exercices_6_a_8() -> None:
    """6 : `Y = X1/3 + X2/2` ; 7 : non ; 8 : rang 2."""
    y = (R(1, 6), R(7, 3), R(17, 6))
    assert coordonnees(y, ((2, 1, 1), (-1, 4, 5))) == (R(1, 3), R(1, 2))
    assert not appartient((0, 0, 1), ((3, 1, 2), (1, 3, 2), (5, 7, 6)))
    assert rang_de_la_famille(((1, 2, -1, 3), (2, 4, 1, -2), (3, 6, 3, -7))) == 2


def test_la_reponse_de_l_exercice_9() -> None:
    """Une base de `R^4`, ou `X = X1/4 + X2/4 + X3/2 + X4/2`."""
    base = ((1, 1, 2, 1), (1, -1, 0, 1), (0, 0, -1, 1), (1, 2, 2, 0))
    assert est_une_base(base)
    assert coordonnees((1, 1, 1, 1), base) == (R(1, 4), R(1, 4), R(1, 2), R(1, 2))


@pytest.mark.parametrize(
    ("famille", "libre", "generatrice"),
    [
        (((1, 1, 0), (0, 1, 1), (1, 3, 2)), False, False),
        (((1, 1, 0), (0, 1, 1), (0, 0, 1)), True, True),
        (((1, 2, 1), (1, 0, 1)), True, False),
        (((1, 1, 1), (2, 1, 0), (2, 0, 1), (0, 0, 1)), False, True),
        (((1, 1, 1), (0, 1, 1), (0, 2, 2), (2, 3, 3)), False, False),
    ],
    ids=["F1", "F2", "F3", "F4", "F5"],
)
def test_la_reponse_de_l_exercice_10(
    famille: tuple[tuple[int, ...], ...], *, libre: bool, generatrice: bool
) -> None:
    """a) libre ? b) generatrice ? c) base si et seulement si les deux."""
    assert est_libre(famille) is libre
    assert est_generatrice(famille) is generatrice
    assert est_une_base(famille) is (libre and generatrice)


def test_les_bases_proposees_a_l_exercice_10() -> None:
    """d) `F3` completee par `(0, 0, 1)` ; e) les trois premiers vecteurs de `F4`."""
    f3 = ((1, 2, 1), (1, 0, 1))
    assert est_une_base((*f3, (0, 0, 1)))
    assert est_une_base(prolonger_en_base(f3))
    f4 = ((1, 1, 1), (2, 1, 0), (2, 0, 1), (0, 0, 1))
    assert extraire_une_base(f4) == f4[:3]
    assert est_une_base(f4[:3])


def test_les_reponses_des_exercices_11_a_13() -> None:
    """11 : meme sous-espace ; 12 : oui ; 13 : `X1 + X3`, puis `(1/3, -1/3, -2/3)`."""
    x = ((1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 0, 1))
    y = ((2, 5, 0, 7), (1, 1, 0, 1), (1, 3, 0, 2))
    assert engendrent_le_meme_sous_espace(x, y)
    assert est_une_base(((1, -1, I), (-1, I, 1), (I, 1, -1)))
    a = ((0, 1, -1), (1, 1, 0), (1, 0, 2))
    assert est_une_base(a)
    assert coordonnees((1, 1, 1), a) == (1, 0, 1)
    b = ((1, 1, 1), (-1, 1, 0), (1, 0, -1))
    assert est_une_base(b)
    assert coordonnees((0, 0, 1), b) == (R(1, 3), -R(1, 3), -R(2, 3))


def test_les_reponses_des_exercices_14_a_17() -> None:
    """Bases, dimensions, somme et intersection."""
    assert extraire_une_base(W1) == W1[:2]
    assert extraire_une_base(W2) == W2
    assert len(somme(W1, W2)) == 3
    assert len(intersection(W1, W2)) == 1
    s = ((1, -1, 2, 1), (2, 0, -1, 0), (0, -2, 5, 1))
    t = ((1, -1, 2, 0), (4, -2, 3, 1), (3, -1, 1, -1))
    dimensions = (
        rang_de_la_famille(s),
        rang_de_la_famille(t),
        len(somme(s, t)),
        len(intersection(s, t)),
    )
    assert dimensions == (3, 3, 3, 3)


def test_la_dimension_de_la_somme_et_de_l_intersection() -> None:
    """`dim(W1 + W2) + dim(W1 inter W2) = dim W1 + dim W2`."""
    gauche = len(somme(W1, W2)) + len(intersection(W1, W2))
    assert gauche == rang_de_la_famille(W1) + rang_de_la_famille(W2)
    commun = intersection(W1, W2)[0]
    assert appartient(commun, W1)
    assert appartient(commun, W2)


def test_la_reponse_de_l_exercice_18() -> None:
    """A : `{(2, 1, 0), (0, 0, 1)}` ; B : `{(1, 2, -3)}` ; C : la base vide."""
    assert base_du_noyau(M([[1, -2, 0]])) == ((2, 1, 0), (0, 0, 1))
    b = base_du_noyau(M([[1, 1, 1], [2, -1, 0]]))
    assert len(b) == 1
    assert engendrent_le_meme_sous_espace(b, ((1, 2, -3),))
    assert base_du_noyau(M([[1, 1, 1], [2, -1, 0], [1, -1, 2]])) == ()


def test_la_reponse_de_l_exercice_19() -> None:
    """`W1 inter W2` a pour base `(-3, 7, -5, 1)`."""
    premier = ((1, 1, 0, -2), (0, -2, 1, 1))
    second = ((1, -2, 1, 0), (2, -3, 0, 1))
    commun = intersection(premier, second)
    assert len(commun) == 1
    assert engendrent_le_meme_sous_espace(commun, ((-3, 7, -5, 1),))


def test_les_sommes_directes_des_exercices_20_et_21() -> None:
    """`R^2 = W1 + W2 = W1 + W3`, et `R^3 = W1 + W2`, sommes directes."""
    assert sont_supplementaires(((2, 1),), ((0, 1),))
    assert sont_supplementaires(((2, 1),), ((1, 1),))
    assert sont_supplementaires(((1, 0, 0),), ((1, 1, 0), (0, 1, 1)))
    assert not sont_supplementaires(((2, 1),), ((4, 2),))
    assert not sont_supplementaires(((1, 0, 0),), ((0, 1, 0),))


def test_une_somme_qui_remplit_l_espace_sans_etre_directe() -> None:
    """Le plan et une de ses droites : la somme est `R^2`, l'intersection non nulle."""
    plan, droite = ((1, 0), (0, 1)), ((1, 1),)
    assert len(somme(plan, droite)) == 2
    assert len(intersection(plan, droite)) == 1
    assert not sont_supplementaires(plan, droite)


def test_un_sous_espace_strictement_inclus_n_est_pas_le_meme() -> None:
    """Une droite du plan, dans un sens puis dans l'autre."""
    plan, droite = ((1, 0, 0), (0, 1, 0)), ((1, 1, 0),)
    assert not engendrent_le_meme_sous_espace(droite, plan)
    assert not engendrent_le_meme_sous_espace(plan, droite)
    assert engendrent_le_meme_sous_espace(plan, ((1, 1, 0), (1, -1, 0)))


def test_une_equation_qui_n_est_pas_polynomiale_n_est_pas_lineaire() -> None:
    """`sin(x1) = 0` : la partie contient `pi` et `pi/2 + pi/2`, mais pas `pi/2`."""
    assert not est_lineaire_homogene([sp.sin(X1)], (X1, X2, X3))


def test_la_reponse_de_l_exercice_22() -> None:
    """`W = {(x, y, z, t) | y = t}` : une seule equation."""
    famille = ((1, 1, 1, 1), (-1, 1, 1, 1), (2, -1, -2, -1))
    obtenues = equations(famille)
    assert len(obtenues) == 1
    assert engendrent_le_meme_sous_espace(obtenues, ((0, 1, 0, -1),))


def test_un_espace_entier_n_a_pas_d_equation() -> None:
    """`n - m = 0` equation quand la famille engendre tout."""
    assert equations(F3) == ()


def test_des_vecteurs_de_tailles_differentes_ne_forment_pas_une_famille() -> None:
    with pytest.raises(ValueError, match="vecteurs de tailles differentes"):
        rang_de_la_famille(((1, 2), (1, 2, 3)))
    with pytest.raises(ValueError, match="famille vide"):
        rang_de_la_famille(())
