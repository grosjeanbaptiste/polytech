"""Mathematiques II : ce que le syllabus ajoute aux planches, par la bibliotheque.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Le
syllabus redit les planches, dont les exemples sont deja refaits ailleurs
(`tests/test_amont_edo_*.py`, `tests/test_amont_series_*.py`). Ces tests ne
portent que sur ce qu'il a en propre.

Source : `CalculDifferentielIntegral2/SYLLABUS/Syllabus_Analyse_2.pdf`.
Les pages sont celles du PDF ; la page imprimee est entre parentheses.
"""

from __future__ import annotations

from fractions import Fraction

import sympy as sp

from polytech.maths.domain.analyse_vectorielle import (
    divergence,
    gradient,
    laplacien,
    rotationnel,
)
from polytech.maths.domain.differentielles.exactes import (
    INCONNUE,
    VARIABLE,
    FormeDifferentielle,
)
from polytech.maths.domain.differentielles.modeles import pendule
from polytech.maths.domain.differentielles.operateurs import D, OperateurDifferentiel
from polytech.maths.domain.differentielles.oscillateurs import gain
from polytech.maths.domain.point_fixe.contraction import iterer

X, Y = VARIABLE, INCONNUE
ESPACE = sp.symbols("x y z", real=True)


def test_la_table_1_1_part_de_un() -> None:
    """Page 9 (7), table 1.1 : `x_0 = 1`, puis les lignes 1 a 4, en fractions."""
    attendus = (Fraction(1), Fraction(3, 2), Fraction(17, 12), Fraction(577, 408))
    iteres = iterer(lambda cote: (cote * cote + 2) / (2 * cote), Fraction(1), 4)
    assert iteres == (*attendus, Fraction(665857, 470832))


def test_le_pendule_et_sa_linearisee_sont_les_relations_9_6_et_9_7() -> None:
    """Page 168 (166) : `theta'' + (g/L) sin(theta) = 0`, puis `sin(theta) ~ theta`."""
    pesanteur, longueur = sp.symbols("g L", positive=True)
    exacte, linearisee = pendule(pesanteur, longueur)
    angle = next(iter(exacte.atoms(sp.Derivative))).expr
    acceleration = angle.diff(angle.args[0], 2)
    assert sp.simplify(exacte.lhs - acceleration) == pesanteur / longueur * sp.sin(
        angle
    )
    assert sp.simplify(linearisee.lhs - acceleration) == pesanteur / longueur * angle


def test_les_racines_de_l_exemple_9_17_sont_celles_de_l_equation_ecrite() -> None:
    """Page 197 (195) : ecart au cours, une coquille dans le polynome.

    L'equation est `y'' + 3 y' + 2 y = ...` et ses racines sont bien -2 et -1.
    Le syllabus ecrit `K(D) = D2 + 3 D + 3`, qui n'a aucune racine reelle.
    """
    assert OperateurDifferentiel(D**2 + 3 * D + 2).racines() == {-2: 1, -1: 1}
    assert sp.discriminant(D**2 + 3 * D + 3, D) == -3
    imprime = OperateurDifferentiel(D**2 + 3 * D + 3).racines()
    assert all(not racine.is_real for racine in imprime)


def test_le_gain_de_l_oscillateur_est_celui_de_la_page_199() -> None:
    """Page 199 (197) : `y'' + y = sin(w t)` donne `sin(w t) / (1 - w2)`, sauf en 1."""
    oscillateur = OperateurDifferentiel(D**2 + 1)
    assert gain(oscillateur, 2) == sp.Rational(1, 3)
    assert gain(oscillateur, sp.Rational(1, 2)) == sp.Rational(4, 3)
    assert gain(oscillateur, 1) == sp.oo


def test_les_deux_identites_de_la_proposition_11_1() -> None:
    """Page 223 (221) : `rot(grad f) = 0`, `div(rot F) = 0` ; le laplacien."""
    scalaire = sp.Function("f")(*ESPACE)
    champ = tuple(sp.Function(nom)(*ESPACE) for nom in "PQR")
    assert rotationnel(gradient(scalaire, ESPACE), ESPACE) == (0, 0, 0)
    assert sp.simplify(divergence(rotationnel(champ, ESPACE), ESPACE)) == 0
    du_gradient = divergence(gradient(scalaire, ESPACE), ESPACE)
    assert sp.simplify(laplacien(scalaire, ESPACE) - du_gradient) == 0


def test_le_potentiel_du_premier_exemple_de_la_section_11_5() -> None:
    """Page 229 (227) : `F = (2xy2 + 3x2, 2x2y)` derive de `x2y2 + x3`."""
    forme = FormeDifferentielle(2 * X * Y**2 + 3 * X**2, 2 * X**2 * Y)
    assert forme.est_exacte()
    assert sp.simplify(forme.potentiel() - (X**2 * Y**2 + X**3)) == 0


def test_le_second_exemple_n_est_conservatif_que_pour_a_4_et_b_un_demi() -> None:
    """Page 230 (228) : `F = (a x3 y + b y2, x4 + x y)`, puis `U = x4 y + x y2 / 2`."""
    a, b = sp.symbols("a b")
    ecart = FormeDifferentielle(
        a * X**3 * Y + b * Y**2, X**4 + X * Y
    ).ecart_de_schwarz()
    coefficients = sp.Poly(ecart, X, Y).coeffs()
    assert sp.solve(coefficients, (a, b)) == {a: 4, b: sp.Rational(1, 2)}
    forme = FormeDifferentielle(4 * X**3 * Y + Y**2 / 2, X**4 + X * Y)
    assert sp.simplify(forme.potentiel() - (X**4 * Y + X * Y**2 / 2)) == 0
