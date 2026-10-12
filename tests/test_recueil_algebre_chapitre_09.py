"""Le chapitre IX du recueil d'algebre, « Les espaces metriques », refait.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, exercices 1 a 33
(pages 141 a 150) ; l'oracle est « Solutions du 9eme chapitre », pages 151
a 153 du syllabus. Le produit scalaire de `C^n` est `somme(x_i conj(y_i))`,
celui que la reponse de l'exercice 2 suppose.

Les deux demonstrations -- 11 et 17 -- sont verifiees ici sur des cas.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.application import recueil_algebre_ch09_fonctions as fonctions
from polytech.maths.application import recueil_algebre_ch09_vecteurs as vecteurs
from polytech.maths.application.macros_recueil_algebre_ch09 import (
    macros_du_chapitre_neuf,
)
from polytech.maths.application.registre import JEUX_DE_MACROS
from polytech.maths.domain.algebre import produits_scalaires as ps

M = sp.Matrix
R = sp.Rational
I = sp.I  # noqa: E741  # l'unite imaginaire, que le syllabus note i
X = fonctions.X


def col(*composantes: object) -> sp.Matrix:
    return M(list(composantes))


def _nul(ecart: object) -> bool:
    simplifie = sp.simplify(ecart)
    if isinstance(simplifie, sp.MatrixBase):
        return bool(simplifie.is_zero_matrix)
    return bool(simplifie == 0)


SCALAIRES = {
    "1NormeUn": sp.sqrt(14),
    "1NormeDeux": sp.sqrt(26),
    "1Distance": sp.sqrt(6),
    "1Produit": sp.Integer(17),
    "2Produit": -8 * I,
    "2Norme": sp.sqrt(14),
    "7b": sp.Integer(3),
    "8": sp.Integer(0),
    "10a": sp.Integer(4),
    "14b": 2 * sp.sqrt(2),
    "16": R(13, 7),
}


@pytest.mark.parametrize("cle", sorted(SCALAIRES))
def test_les_nombres_des_exercices_1_a_16(cle: str) -> None:
    assert _nul(vecteurs.nombres()[cle] - SCALAIRES[cle])


def test_exercice_15_la_distance_a_une_droite() -> None:
    """Le syllabus donne `~ 3,74`."""
    exacte = vecteurs.nombres()["15"]
    assert float(exacte) == pytest.approx(3.74, abs=5e-3)


def test_exercice_3_base_orthogonale_unitaire() -> None:
    assert vecteurs.exercice_3() == {"a": True, "b": False, "c": False}


RC = sp.sqrt
ORTHONORMALES = {
    "3d": (
        col(1, 1, 0, 1) / RC(3),
        col(4, -5, 0, 1) / RC(42),
        col(-4, -2, -7, 6) / RC(105),
    ),
    "4a": (col(1, I, 0) / RC(2), col((1 + I) / 2, (1 - I) / 2, 1) / RC(2)),
    "4b": (col(1, -1, -I) / RC(3), col(1, 2 + 3 * I, 3 - I) / (2 * RC(6))),
    "5a": (col(1, 1, -1) / RC(3), col(1, 0, 1) / RC(2)),
    "5b": (col(2, 1, 1) / RC(6), col(-1, 7, -5) / (5 * RC(3))),
    "6": (col(1, 2, 1, 0) / RC(6), col(-1, -2, 5, 3) / RC(39)),
    "7a": (
        col(1, 1, 0, 0) / RC(2),
        col(1, -1, 1, 1) / 2,
        col(-RC(2) / 3, RC(2) / 3, RC(2) / 2, RC(2) / 6),
    ),
    "13": (col(0, 0, I, -I) / RC(2), col(-1, I, 0, 0) / RC(2)),
}


@pytest.mark.parametrize("cle", sorted(ORTHONORMALES))
def test_les_bases_orthonormales(cle: str) -> None:
    calculee = vecteurs.orthonormales()[cle]
    assert len(calculee) == len(ORTHONORMALES[cle])
    for trouve, attendu in zip(calculee, ORTHONORMALES[cle], strict=True):
        assert _nul(trouve - attendu)
        assert _nul(ps.norme(trouve, ps.usuel) - 1)


def test_exercice_9_les_vecteurs_de_projection_donnee() -> None:
    """Le syllabus : `(2 - i alpha, alpha, beta, beta - 2)`."""
    projete, directions = vecteurs.exercice_9()
    assert projete == col(1, -I, 1, -1)
    alpha, beta = sp.symbols("alpha beta")
    attendu = col(2 - I * alpha, alpha, beta, beta - 2)
    ecart = (attendu - projete).subs({alpha: 0, beta: 0})
    assert M.hstack(*directions, ecart).rank() == len(directions)
    derivees = M.hstack(sp.diff(attendu, alpha), sp.diff(attendu, beta))
    assert M.hstack(*directions, derivees).rank() == len(directions)


def test_exercice_10_l_orthogonal_dans_r6() -> None:
    base = vecteurs.exercice_10()
    generateurs = M([[1, 1, -2, 3, 4, 5], [0, 0, 1, 1, 0, 7]])
    assert all((generateurs * v).is_zero_matrix for v in base)


@pytest.mark.parametrize("dimension", [2, 3, 5])
def test_exercice_11_l_orthogonal_de_l_hyperplan(dimension: int) -> None:
    taille, orthogonal = vecteurs.exercice_11(dimension)
    assert taille == dimension - 1
    assert orthogonal == (col(*[1] * dimension),)


def test_exercice_12_l_orthogonal_dans_c3() -> None:
    (base,) = vecteurs.exercice_12()
    assert M.hstack(base, col(3, -4, -I)).rank() == 1


def test_exercice_14_la_projection() -> None:
    assert vecteurs.exercice_14() == col(1, -I, 1, -1)


def test_exercice_17_la_trace_donne_un_produit_scalaire() -> None:
    a, b = (
        M(2, 2, sp.symbols("a1:5", real=True)),
        M(2, 2, sp.symbols("b1:5", real=True)),
    )
    assert sp.expand(ps.de_la_trace(a, b) - ps.de_la_trace(b, a)) == 0
    assert sp.expand(ps.de_la_trace(a, a) - sum(x**2 for x in a)) == 0


MATRICES = {
    "18a": (
        M([[1, 0], [0, 0]]),
        M([[0, 1], [1, 1]]) / RC(3),
        M([[0, -1], [-1, 2]]) / RC(6),
    ),
    "18b": M([[1, R(5, 2)], [R(5, 2), 4]]),
    "19": 1 / RC(2),
    "20Orthogonal": (M([[-2, 2], [1, 0]]), M([[-1, -1], [0, 1]])),
    "20SurOrthogonal": M([[-2, 2], [1, 0]]) / 9,
    "20SurW": M([[R(11, 9), -R(2, 9)], [R(26, 9), 1]]),
    "21Produit": R(2, 3),
    "21Norme": RC(R(2, 3)),
    "22": (sp.Integer(1), X, X**2 - R(1, 3)),
    "23": (sp.Integer(1), X, X**2 - 2),
    "24": (X**2 - R(9, 5) * X - R(3, 5),),
    "25": (sp.Integer(1), X, X**2 - R(1, 3)),
    "26": (
        1 / RC(2 * sp.pi),
        sp.sin(X) / RC(sp.pi),
        sp.cos(X) / RC(sp.pi),
        sp.sin(5 * X) / RC(sp.pi),
    ),
    "27a": (sp.E - 1 / sp.E) / 2 + 3 * X / sp.E,
    "27b": RC(1 - 7 / sp.E**2),
    "28": sp.pi - 2 * sp.sin(X),
    "29": 1 + sp.pi - 2 * sp.sin(X) - sp.sin(2 * X),
    "30": (R(1, 3), R(1, 3), R(1, 3)),
    "31": (R(9, 10), R(6, 5)),
    "32": 4 * RC(R(2, 7)),
    "33Projection": M([[7, 13], [1, -7]]) / 4,
    "33SurOrthogonal": M([[-3, 3], [3, 3]]) / 4,
}


@pytest.mark.parametrize("cle", sorted(MATRICES))
def test_exercices_18_a_33(cle: str) -> None:
    calculee, attendue = fonctions.reponses()[cle], MATRICES[cle]
    if isinstance(attendue, tuple):
        assert len(calculee) == len(attendue)
        assert all(_nul(c - a) for c, a in zip(calculee, attendue, strict=True))
        return
    assert _nul(calculee - attendue)


def test_exercice_26_la_famille_est_orthogonale() -> None:
    assert fonctions.exercice_26_est_orthogonale() is True


def test_exercice_33_la_dimension_selon_a() -> None:
    generique, particuliers = fonctions.dimension_du_33()
    assert generique == 3
    assert particuliers == {sp.Integer(-3): 2}


def test_exercice_33_hors_de_moins_trois_la_matrice_est_deja_dans_w() -> None:
    """`W` est alors un hyperplan qui contient `A` : la projection est `A`."""
    assert fonctions.reponses()["33Generique"] == M([[1, 4], [1, -1]])


def test_une_dimension_non_strictement_positive_est_refusee() -> None:
    with pytest.raises(ValueError, match="dimension non strictement positive : "):
        vecteurs.exercice_11(0)


def test_les_etapes_de_gram_schmidt_sont_orthogonales() -> None:
    for famille in vecteurs.orthogonales().values():
        assert ps.est_orthogonale(famille, ps.usuel)


def test_la_droite_du_31_passe_par_ses_deux_extremites() -> None:
    (x0, y0), (x1, y1) = fonctions.extremites_du_31()
    pente, ordonnee = fonctions.reponses()["31"]
    assert (x0, x1) == (0, 4)
    assert y0 == ordonnee
    assert y1 == 4 * pente + ordonnee


def test_le_jeu_de_macros_du_chapitre_neuf() -> None:
    macros = macros_du_chapitre_neuf()
    assert all(nom.startswith("AlgNeuf") and nom.isalpha() for nom in macros)
    assert all(valeur.strip() for valeur in macros.values())
    assert macros["AlgNeufUnProduit"] == "17"
    assert macros["AlgNeufTrenteUnPente"] == r"\frac{9}{10}"
    assert macros["AlgNeufTrenteUnOrdonnee"] == r"\frac{6}{5}"
    assert macros["AlgNeufTrente"].startswith(r"\left(\frac{1}{3},")
    assert macros["AlgNeufQuinzeApprochee"] == "3.738"
    assert macros["AlgNeufTrenteUnPoints"] == "(0,1) (1,3) (2,2) (3,4) (4,5)"
    assert macros["AlgNeufTrenteUnDroite"] == "(0,1.2) (4,4.8)"
    assert r"\sqrt{105}" in macros["AlgNeufTroisD"]
    assert r"\frac{7}{3}" in macros["AlgNeufCinqBOrthogonale"]
    assert "sqrt" not in macros["AlgNeufCinqBOrthogonale"]
    assert macros["AlgNeufVingtDeux"].startswith(r"\left\{1,\; x,")
    assert "algebre-09-espaces-metriques" in JEUX_DE_MACROS
