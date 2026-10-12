"""Le chapitre VIII du recueil d'algebre, section VIII.4 : exercices 1 a 13.

Source : `Mathematique/SYLLABUS/Syllabus_Algebre.pdf`, pages 102 a 104 ;
l'oracle est « Solutions du 8eme chapitre », pages 124 et 125 du syllabus.
Lineaire ou non, noyau, image, inverse, rang.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.application import recueil_algebre_ch08a as partie
from polytech.maths.application import recueil_algebre_ch08a_suite as suite
from polytech.maths.application.macros_recueil_algebre_ch08a import (
    macros_de_la_partie_a,
)
from polytech.maths.application.registre import JEUX_DE_MACROS
from polytech.maths.domain.algebre.familles_de_vecteurs import est_une_base
from polytech.maths.domain.algebre.linearite import respecte_les_combinaisons
from polytech.maths.domain.algebre.sous_espaces import engendrent_le_meme_sous_espace

R = sp.Rational
X, Y, Z = sp.symbols("x y z")
A, B, C = sp.symbols("a b c")
T = suite.PARAMETRE


def test_exercice_1_lineaire_ou_non() -> None:
    assert partie.exercice_1() == {"a": True, "b": False, "c": False}


@pytest.mark.parametrize(
    ("lettre", "attendues"),
    [
        ("a", ((-1, -1), (4, 5))),
        ("b", ((-R(2, 3), 1), (R(11, 3), -3))),
        ("c", ((-2, -1), (4, 2))),
    ],
)
def test_exercice_2_les_images_de_la_base_canonique(
    lettre: str, attendues: tuple
) -> None:
    assert partie.exercice_2()[lettre] == attendues


def test_exercice_3_quatre_applications_sur_les_polynomes() -> None:
    """`f2` n'est lineaire que si `q = 0` : on l'eprouve avec `q = 1`, puis `q = 0`."""
    assert partie.exercice_3() == {"f1": True, "f2": False, "f3": True, "f4": True}
    p, q = sp.Function("p")(X), sp.Function("q")(X)
    assert respecte_les_combinaisons(lambda u: u + 0, p, q)


def test_exercice_4_l_evaluation_et_la_conjugaison() -> None:
    assert partie.exercice_4() == {"a": True, "b": True}


def test_exercice_5a_noyau_et_image_d_une_projection() -> None:
    application = partie.APPLICATIONS["5a"]
    assert application.noyau() == ((1, 0, 0, 0), (0, 0, 1, 0))
    assert application.est_surjective()


def test_exercice_5c_la_partie_impaire() -> None:
    """Le noyau contient les paires, l'image est faite d'impaires."""
    paire, impaire = sp.cos(X), sp.sin(X)
    assert partie.partie_impaire(paire) == 0
    assert sp.simplify(partie.partie_impaire(impaire) - impaire) == 0
    quelconque = partie.partie_impaire(sp.exp(X))
    assert sp.simplify(quelconque.subs(X, -X) + quelconque) == 0


def test_exercice_6_aucune_application_lineaire_ne_convient() -> None:
    existe, relation = partie.exercice_6()
    assert existe is False
    assert relation == (1, -2, 1)


def test_exercice_7_l_image_d_un_vecteur_quelconque() -> None:
    assert partie.exercice_7() == (X, X, X, X + Y + Z)


@pytest.mark.parametrize("valeur", [0, 1, sp.Symbol("a")])
def test_exercice_8_le_noyau_de_la_derivation_decalee(valeur: sp.Expr) -> None:
    solution = partie.noyau_de_la_derivation_moins(valeur)
    assert sp.simplify(sp.diff(solution, X) - valeur * solution) == 0
    assert solution.free_symbols >= {partie.CONSTANTE}


@pytest.mark.parametrize(
    ("lettre", "attendue"),
    [
        ("a", ((A + B) / 2, (A - B) / 2)),
        ("b", ((5 * A + B) / 13, (3 * A - 2 * B) / 13)),
        ("c", None),
        ("d", (A + 2 * B - C, -A - B + C, -2 * A - 5 * B + 3 * C)),
    ],
)
def test_exercice_9_les_inverses(lettre: str, attendue: tuple | None) -> None:
    inverse = partie.exercice_9()[lettre]
    if attendue is None:
        assert inverse is None
        return
    assert inverse is not None
    ecart = sp.Matrix(list(inverse)) - sp.Matrix(list(attendue))
    assert sp.simplify(ecart).is_zero_matrix


@pytest.mark.parametrize(("degre", "ordre"), [(3, 1), (3, 2), (3, 4), (3, 6), (4, 5)])
def test_exercice_10_noyau_et_image_des_derivees_successives(
    degre: int, ordre: int
) -> None:
    """`Ker D^k = R[X]_(k-1)` et `Im D^k = R[X]_(n-k)`, tant que `k <= n`."""
    noyau, image = suite.exercice_10(degre, ordre)
    assert noyau == min(ordre, degre + 1)
    assert image == max(degre + 1 - ordre, 0)


@pytest.mark.parametrize("ordre", [2, 3])
def test_exercice_11_parties_symetrique_et_antisymetrique(ordre: int) -> None:
    dimensions = suite.exercice_11(ordre)
    symetriques, antisymetriques = ordre * (ordre + 1) // 2, ordre * (ordre - 1) // 2
    assert dimensions["f1"] == (antisymetriques, symetriques)
    assert dimensions["f2"] == (symetriques, antisymetriques)


@pytest.mark.parametrize("ordre", [2, 3, 4])
def test_exercice_12_les_matrices_de_trace_nulle(ordre: int) -> None:
    assert suite.exercice_12(ordre) == ordre**2 - 1


def test_exercice_13_le_rang_selon_t() -> None:
    generique, particuliers = suite.exercice_13()
    assert generique == 3
    assert particuliers == {sp.Integer(-1): 1, sp.Integer(1): 2}


def test_exercice_13_les_bases_de_l_image() -> None:
    bases = suite.images_du_13()
    assert engendrent_le_meme_sous_espace(bases[-1], ((1, -1, 1),))
    assert engendrent_le_meme_sous_espace(bases[1], ((1, 1, 1), (1, 1, -1)))
    completees = suite.completees_du_13()
    assert all(est_une_base(b) for b in completees.values())
    assert est_une_base(((1, -1, 1), (0, 1, 0), (0, 0, 1)))
    assert est_une_base(((1, 1, 1), (1, 1, -1), (0, 1, 0)))


def test_un_ordre_non_strictement_positif_est_refuse() -> None:
    with pytest.raises(ValueError, match="ordre non strictement positif : 0"):
        suite.exercice_12(0)
    with pytest.raises(ValueError, match="ordre non strictement positif : 0"):
        suite.exercice_11(0)
    with pytest.raises(ValueError, match="ordre non strictement positif : -1"):
        suite.exercice_10(3, -1)


def test_exercice_1a_le_carre_unite_devient_un_parallelogramme() -> None:
    """La figure du corrige : les sommets du carre, puis leurs images."""
    carre, image = partie.carre_et_son_image()
    assert carre == ((0, 0), (1, 0), (1, 1), (0, 1))
    assert image == ((0, 0), (2, 0), (3, 1), (1, 1))


def test_exercice_9_les_determinants_decident_de_l_inverse() -> None:
    assert partie.determinants_du_9() == {"a": -2, "b": -13, "c": 0, "d": 1}


def test_le_jeu_de_macros_de_la_partie_a() -> None:
    macros = macros_de_la_partie_a()
    assert all(nom.startswith("AlgHuit") and nom.isalpha() for nom in macros)
    assert all(valeur.strip() for valeur in macros.values())
    assert macros["AlgHuitSept"] == r"\left(x,\, x,\, x,\, x + y + z\right)"
    assert macros["AlgHuitNeufCDet"] == "0"
    assert macros["AlgHuitDeuxAUn"] == r"\left(-1,\, -1\right)"
    assert macros["AlgHuitDeuxADeux"] == r"\left(4,\, 5\right)"
    assert "AlgHuitNeufC" not in macros
    assert macros["AlgHuitCarre"] == "(0,0) (1,0) (1,1) (0,1)"
    assert "algebre-08a-applications-lineaires" in JEUX_DE_MACROS
