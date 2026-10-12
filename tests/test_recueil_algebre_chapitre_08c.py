"""Le chapitre VIII du recueil d'algebre, section VIII.8 : treize sujets d'examen.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, exercices 33 a 45
(pages 117 a 123) ; l'oracle est « Solutions du 8eme chapitre », pages 128
a 134 du syllabus. Une base ne s'ecrit pas d'une seule facon : celle du
syllabus est eprouvee -- elle engendre le meme sous-espace que celle que
la bibliotheque trouve.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.application import recueil_algebre_ch08c_enonces as enonces
from polytech.maths.application import recueil_algebre_ch08c_etude as etude
from polytech.maths.application import recueil_algebre_ch08c_metrique as metrique
from polytech.maths.application.macros_recueil_algebre_ch08c import (
    macros_de_la_partie_c,
)
from polytech.maths.application.registre import JEUX_DE_MACROS
from polytech.maths.domain.algebre.sous_espaces import engendrent_le_meme_sous_espace

M = sp.Matrix
R = sp.Rational
X = enonces.X
T, A, B, AL, BE = enonces.T, enonces.A, enonces.B, enonces.ALPHA, enonces.BETA

MATRICES = {
    33: [[2, 1, -2], [2, 2, -2], [0, 1, 0], [0, 0, 0]],
    35: [[2 * T, 1, 1, 0], [1, 2 * T, 0, 1], [1, 0, 2 * T, 1], [0, 1, 1, 2 * T]],
    36: [[A + B, 0, 0, 0], [0, A, B, 0], [0, B, A, 0], [0, 0, 0, A + B]],
    37: [
        [0, BE, 0, AL],
        [AL, 0, BE, 0],
        [0, 1, 0, 1],
        [AL, 0, BE, 0],
        [0, AL, 0, BE],
        [BE, 0, AL, 0],
    ],
    38: [
        [B, 0, A, 0, A, 0],
        [0, A, 0, 1, 0, B],
        [A, 0, B, 0, B, 0],
        [0, B, 0, 1, 0, A],
    ],
    39: [[1, 1, 1, 1, 1], [0, 1, 2, 3, 4], [1, -1, 1, -1, 1], [0, 1, -2, 3, -4]],
    40: [[4, 2, -2, -4], [-1, 1, 4, -4]],
    41: [[-1, 0, 0, 1], [-1, 1, 1, 0], [-1, 1, 1, 0], [-1, 0, 0, 1]],
    43: [[0, A - 2, A, 2 - A], [1, 1, 2, -1], [0, A - 2, 0, 2 - A]],
    44: [[0, 1, 2, 0], [1, 0, 0, -1], [0, 1, 1, 1]],
}


@pytest.mark.parametrize("numero", sorted(MATRICES))
def test_la_matrice_de_chaque_sujet(numero: int) -> None:
    assert sp.simplify(etude.matrice(numero) - M(MATRICES[numero])).is_zero_matrix


NOYAUX = {
    (33, ""): ((1, 0, 1),),
    (34, ""): ((1, 0, -1, 0),),
    (35, "TUn"): ((1, -1, -1, 1),),
    (35, "TMoinsUn"): ((1, 1, 1, 1),),
    (35, "TZero"): ((1, 0, 0, -1), (0, 1, -1, 0)),
    (36, "Opposes"): ((1, 0, 0, 0), (0, 1, 1, 0), (0, 0, 0, 1)),
    (36, "Egaux"): ((0, 1, -1, 0),),
    (37, "Egaux"): ((1, 0, -1, 0), (0, 1, 0, -1)),
    (37, "Opposes"): ((1, 0, 1, 0),),
    (38, "Egaux"): (
        (-1, 0, 1, 0, 0, 0),
        (-1, 0, 0, 0, 1, 0),
        (0, 1, 0, -A, 0, 0),
        (0, 0, 0, -A, 0, 1),
    ),
    (38, "Opposes"): ((1, 0, 1, 0, 0, 0), (1, 0, 0, 0, 1, 0), (0, 1, 0, 0, 0, 1)),
    (38, ""): ((0, 1, 0, -(A + B), 0, 1), (0, 0, -1, 0, 1, 0)),
    (39, ""): ((1, 0, -2, 0, 1),),
    (40, ""): ((5, -7, 3, 0), (-2, 10, 0, 3)),
    (41, ""): ((1, 0, 1, 1), (0, 1, -1, 0)),
    (43, "AZero"): ((-2, 0, 1, 0), (0, 1, 0, 1)),
    (43, "ADeux"): ((1, 0, 0, 1), (0, 1, 0, 1)),
    (43, ""): ((0, 1, 0, 1),),
    (44, ""): ((1, -2, 1, 1),),
    (45, ""): ((0, 1, 1, 0),),
}
"""Exercice et cas : la base du noyau que le syllabus imprime."""


@pytest.mark.parametrize("cas", sorted(NOYAUX))
def test_chaque_noyau(cas: tuple[int, str]) -> None:
    calcule = etude.noyau(*cas)
    assert len(calcule) == len(NOYAUX[cas])
    assert engendrent_le_meme_sous_espace(calcule, NOYAUX[cas])


@pytest.mark.parametrize("cas", [(35, ""), (36, ""), (37, "")])
def test_hors_des_cas_singuliers_le_noyau_est_nul(cas: tuple[int, str]) -> None:
    assert etude.noyau(*cas) == ()


IMAGES = {
    (33, ""): ((1, 1, 0, 0), (1, 2, 1, 0)),
    (34, ""): ((1, 0, 0, 0), (1, 2, 0, 0), (0, 1, 1, 1)),
    (35, "TUn"): ((2, 1, 1, 0), (1, 2, 0, 1), (1, 0, 2, 1)),
    (35, "TMoinsUn"): ((-2, 1, 1, 0), (1, -2, 0, 1), (1, 0, -2, 1)),
    (35, "TZero"): ((0, 1, 1, 0), (1, 0, 0, 1)),
    (36, "Opposes"): ((0, 1, -1, 0),),
    (36, "Egaux"): ((1, 0, 0, 0), (0, 1, 1, 0), (0, 0, 0, 1)),
    (37, "Egaux"): ((0, AL, 0, AL, 0, AL), (AL, 0, 1, 0, AL, 0)),
    (37, "Opposes"): (
        (0, AL, 0, AL, 0, -AL),
        (-AL, 0, 1, 0, AL, 0),
        (AL, 0, 1, 0, -AL, 0),
    ),
    (38, "Egaux"): ((1, 0, 1, 0), (0, 1, 0, 1)),
    (38, "Opposes"): ((-1, 0, 1, 0), (0, 1, 0, -1), (0, 1, 0, 1)),
    (41, ""): ((1, 0, 0, 1), (0, 1, 1, 0)),
    (43, "AZero"): ((0, 1, 0), (1, 0, 1)),
    (43, "ADeux"): ((1, 0, 0), (0, 1, 0)),
}
"""Exercice et cas : la base de l'image que le syllabus imprime."""


@pytest.mark.parametrize("cas", sorted(IMAGES))
def test_chaque_image(cas: tuple[int, str]) -> None:
    calculee = etude.image(*cas)
    assert len(calculee) == len(IMAGES[cas])
    assert engendrent_le_meme_sous_espace(calculee, IMAGES[cas])


RANGS_PLEINS = {35: 4, 36: 4, 37: 4, 38: 4, 39: 4, 43: 3, 44: 3, 45: 3}


@pytest.mark.parametrize("numero", sorted(RANGS_PLEINS))
def test_les_rangs_pleins(numero: int) -> None:
    assert etude.rang(numero) == RANGS_PLEINS[numero]


def test_les_determinants_donnent_les_cas_singuliers() -> None:
    assert sp.expand(etude.determinant(35) - 16 * T**2 * (T - 1) * (T + 1)) == 0
    assert sp.expand(etude.determinant(36) - (A - B) * (A + B) ** 3) == 0


def test_exercice_33_aucun_antecedent() -> None:
    assert etude.antecedents(33, (8, 21, 4, 0)) == sp.EmptySet


def test_exercice_36_l_inverse() -> None:
    """`f^-1(Y) = (a Y - b Y^t) / (a^2 - b^2)`."""
    y = M(2, 2, sp.symbols("y1:5"))
    attendu = (A * y - B * y.T) / (A**2 - B**2)
    calcule = etude.inverse_du_36() * M(list(y))
    assert sp.simplify(calcule - M(list(attendu))).is_zero_matrix


def test_exercice_39_les_antecedents() -> None:
    """Le syllabus : `(-1 + a4) + X - 2 a4 X^2 + a4 X^4`."""
    (solution,) = etude.antecedents(39, (0, 1, -2, 1))
    (libre,) = sp.Matrix(list(solution)).free_symbols
    attendu = (libre - 1, 1, -2 * libre, 0, libre)
    assert tuple(sp.expand(s) for s in solution) == attendu


def test_exercice_40_les_antecedents() -> None:
    (solution,) = etude.antecedents(40, (1, 3))
    u3, u4 = solution[2], solution[3]
    assert sp.simplify(solution[0] + (5 - 10 * u3 + 4 * u4) / 6) == 0
    assert sp.simplify(solution[1] - (13 - 14 * u3 + 20 * u4) / 6) == 0


def test_exercice_41_les_valeurs_qui_donnent_des_antecedents() -> None:
    """`3 - alpha X + 5 X^2 + beta X^3` : `alpha = -5`, `beta = 3`."""
    assert etude.antecedents(41, (3, -AL, 5, BE)) == sp.EmptySet
    (solution,) = etude.antecedents(41, (3, 5, 5, 3))
    a0, a1 = solution[0], solution[1]
    assert sp.simplify(solution[2] - (5 + a0 - a1)) == 0
    assert sp.simplify(solution[3] - (a0 + 3)) == 0


ORTHOGONAUX = {
    (34, ""): ((1, 0, 1, 0), (0, 1, 0, 0), (0, 0, 0, 1)),
    (36, "Opposes"): ((0, 1, -1, 0),),
    (36, "Egaux"): ((1, 0, 0, 0), (0, 1, 1, 0), (0, 0, 0, 1)),
    (37, "Egaux"): ((1, 0, 1, 0), (0, 1, 0, 1)),
    (37, "Opposes"): ((1, 0, -1, 0), (0, 1, 0, 0), (0, 0, 0, 1)),
    (41, ""): ((-1, 1, 1, 0), (-1, 0, 0, 1)),
}


@pytest.mark.parametrize("cas", sorted(ORTHOGONAUX))
def test_l_orthogonal_du_noyau_pour_la_trace(cas: tuple[int, str]) -> None:
    calcule = metrique.orthogonal_du_noyau(*cas)
    assert engendrent_le_meme_sous_espace(calcule, ORTHOGONAUX[cas])
    for vecteur in etude.noyau(*cas):
        assert all(sp.simplify(M(list(o)).dot(M(list(vecteur)))) == 0 for o in calcule)


def test_exercice_33_orthogonal_et_projections() -> None:
    a, b, c = sp.symbols("a b c")
    orthogonal, calcule_noyau, calcule_orthogonal = metrique.exercice_33()
    lignes = [sp.Poly(p, X).all_coeffs()[::-1] for p in orthogonal]
    completees = tuple(tuple(ligne + [0] * (3 - len(ligne))) for ligne in lignes)
    assert engendrent_le_meme_sous_espace(completees, ((1, 0, -R(5, 2)), (0, 1, 0)))
    sur_le_noyau = (16 * a + 40 * c) / 56 * (1 + X**2)
    sur_l_orthogonal = b * X + (-2 * a + 2 * c) / 14 * (2 - 5 * X**2)
    assert sp.expand(calcule_noyau - sur_le_noyau) == 0
    assert sp.expand(calcule_orthogonal - sur_l_orthogonal) == 0


def test_exercice_41_les_polynomes_orthogonaux_a_x() -> None:
    for polynome in metrique.exercice_41():
        assert sp.integrate(polynome * X, (X, -1, 1)) == 0
    assert len(metrique.exercice_41()) == 3


def test_exercice_42_la_composee() -> None:
    g, image, orthogonal = metrique.exercice_42()
    assert g.rank() == 1
    assert image == (-5, 13, 5)
    assert engendrent_le_meme_sous_espace(orthogonal, ((1, 0, 0, 1),))
    assert tuple(etude.matrice(42) * M(sp.symbols("x y z"))) == (
        sp.Symbol("x") - 2 * sp.Symbol("y") - 2 * sp.Symbol("z"),
        2 * sp.Symbol("x") + sp.Symbol("y") + 5 * sp.Symbol("z"),
        4 * sp.Symbol("x") - 3 * sp.Symbol("y") + 2 * sp.Symbol("z"),
    )


def test_exercice_44_une_base_de_la_somme() -> None:
    attendue = ((1, -2, 1, 1), (1, 0, 1, -1), (0, 1, 0, 1))
    assert engendrent_le_meme_sous_espace(metrique.exercice_44(), attendue)
    assert len(metrique.exercice_44()) == 3
    w = tuple(tuple(v) for v in metrique.W_44.nullspace())
    assert engendrent_le_meme_sous_espace(w, ((0, 1, 0, 1), (1, 0, 1, -1)))


def test_exercice_45() -> None:
    egalite, symetriques, associee = metrique.exercice_45()
    assert egalite is False
    attendues = ((1, 0, 0, 0), (0, 1, 1, 0), (0, 0, 0, 1))
    assert engendrent_le_meme_sous_espace(symetriques, attendues)
    attendue = M(
        [
            [-R(1, 2), -1, -R(1, 2), -R(1, 2)],
            [R(3, 2), 1, R(1, 2), R(3, 2)],
            [R(5, 6), R(2, 3), R(1, 6), R(5, 6)],
        ]
    )
    assert associee == attendue


def test_un_exercice_ou_un_cas_inconnu_est_refuse() -> None:
    with pytest.raises(KeyError, match="dans la section VIII"):
        etude.matrice(12)
    with pytest.raises(KeyError, match=" : pas de cas "):
        etude.noyau(35, "Absent")


def test_le_jeu_de_macros_de_la_partie_c() -> None:
    macros = macros_de_la_partie_c()
    assert all(nom.startswith("AlgHuit") and nom.isalpha() for nom in macros)
    assert all(valeur.strip() for valeur in macros.values())
    assert macros["AlgHuitTrenteTroisNoyau"] == r"\left\{X^{2} + 1\right\}"
    assert "AlgHuitTrenteCinqNoyau" not in macros
    assert "AlgHuitTrenteCinqImage" not in macros
    assert r"\begin{matrix}" in macros["AlgHuitTrenteCinqTZeroNoyau"]
    assert macros["AlgHuitQuaranteDeuxImage"] == r"\left(-5,\, 13,\, 5\right)"
    assert "algebre-08c-examens" in JEUX_DE_MACROS
