"""Le recueil de primitives du syllabus de 2011, exercice 1 : cinquante-six primitives.

Source : `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf`, chapitre
« Primitives », pages 60 et 61 du PDF ; l'oracle est « Solutions du
chapitre XII », pages 64 a 67. Le document est un scan : enonces et
solutions ont ete lus sur les pages.

Une primitive ne s'ecrit pas d'une seule facon. Deux controles, donc, et
aucun ne compare des ecritures : la derivee de la primitive rend
l'integrande, et la reponse du syllabus aussi.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.application import recueil_primitives_enonces as enonces
from polytech.maths.application import recueil_primitives_gestes as gestes
from polytech.maths.application import recueil_primitives_presentation as presentation
from polytech.maths.application import recueil_primitives_un as un
from polytech.maths.application.macros_recueil_primitives_un import (
    macros_de_l_exercice_un,
)
from polytech.maths.application.registre import JEUX_DE_MACROS

X, A, B, P, N = enonces.X, enonces.A, enonces.B, enonces.P, enonces.N
R = sp.Rational
ln, sin, cos, tan, exp, sqrt = sp.log, sp.sin, sp.cos, sp.tan, sp.exp, sp.sqrt
atan, asin, sinh, cosh, tanh, asinh = (
    sp.atan,
    sp.asin,
    sp.sinh,
    sp.cosh,
    sp.tanh,
    sp.asinh,
)

D = 16**2 + 30**2
SOLUTIONS = {
    1: X + 2 * X**2 + R(4, 3) * X**3 + R(2, 5) * X**5 + R(2, 3) * X**6 + X**9 / 9,
    2: -exp(-X) * (5 * X**2 + 10 * X + 11),
    3: 2 / (3 * B) * (A + B * X) ** R(3, 2),
    4: 2 / B**2 * ((A + B * X) ** R(5, 2) / 5 - A / 3 * (A + B * X) ** R(3, 2)),
    5: atan(sqrt(R(3, 7)) * X) / sqrt(21),
    6: -ln(sp.Abs(cos(X))),
    7: -2 * sqrt(2) * cos(X / 2),
    8: 2 * sqrt(2) * sin(X / 2),
    9: 3 ** (5 * X) / ln(3**5),
    10: atan(exp(X)),
    11: -(cos(X) ** (N + 1)) / (N + 1),
    12: tan(X) - X,
    13: R(1, 16) * ((5 + 4 * X) ** 20 / 20 - R(5, 19) * (5 + 4 * X) ** 19),
    14: -ln(sp.Abs(cos(ln(X)))),
    15: -sp.cot(X),
    16: P**2 / (P**2 + A**2) * exp(A * X) * (sin(P * X) / P + A * cos(P * X) / P**2),
    17: ln(5 * X**2 + 18) / 10,
    18: (3 + 10 * exp(X)) ** R(6, 5) / 12,
    19: P**2 / (P**2 + A**2) * exp(A * X) * (-cos(P * X) / P + A * sin(P * X) / P**2),
    20: -ln(sp.Abs(exp(-P * X) + B)) / P,
    21: (X - sin(2 * X) / 2) / 2,
    22: 2 * exp(X / 2) * (X**4 - 8 * X**3 + 48 * X**2 - 192 * X + 384),
    23: X * cosh(X) - sinh(X),
    24: tanh(X),
    25: R(100, 101) * exp(X / 2) * (-cos(5 * X) / 5 + sin(5 * X) / 50),
    26: X * (sin(ln(X)) - cos(ln(X))) / 2,
    27: (sinh(2 * X) / 2 - X) / 2,
    28: ln(sp.Abs(X**2 + 4 * X + 5)) / 2 - atan(X + 2),
    29: 2 * exp(sqrt(X)) * (sqrt(X) - 1),
    30: X * atan(X) - ln(1 + X**2) / 2,
    31: X ** (P + 1) / (P + 1) * (ln(A * X) - 1 / (P + 1)),
    32: X**3 / 6 + X**2 / 4 * sin(2 * X) + X / 4 * cos(2 * X) - sin(2 * X) / 8,
    33: X * asin(X) + sqrt(1 - X**2),
    34: 2 * sqrt(1 + exp(X)),
    35: R(1, 3) * (1 + X**3) ** R(77, 50) / R(77, 50),
    36: 2 * sqrt(3) / 3 * atan(2 / sqrt(3) * (X + R(1, 2))),
    37: sin(X) * (3 * (X + 1) ** 2 - 6) - cos(X) * ((X + 1) ** 3 - 6 * (X + 1)),
    38: 4 * (sqrt(X + 1) / 2 + (X + 1) ** R(1, 4) + ln(sp.Abs((X + 1) ** R(1, 4) - 1))),
    39: (X**3 / 3 + X) * atan(X) - X**2 / 6 - ln(1 + X**2) / 3,
    40: exp(-5 * X)
    / 34
    * (
        (-5 * cos(3 * X) + 3 * sin(3 * X)) * (R(3, 17) * X + R(60, D))
        + (-3 * cos(3 * X) - 5 * sin(3 * X)) * (X**2 + R(5, 17) * X + R(32, D))
    ),
    41: (sinh(4 * X) - 2 * sinh(2 * X)) / 8,
    42: (X + 1) ** 3 * sinh(X)
    - 3 * (X + 1) ** 2 * cosh(X)
    + 6 * (X + 1) * sinh(X)
    - 6 * cosh(X),
    43: atan(X / 2) / 2,
    44: asin(sqrt(3) * X) / sqrt(3),
    46: X**5 / 5 * (ln(X) ** 3 - R(3, 5) * ln(X) ** 2 + R(6, 25) * ln(X) - R(6, 125)),
    47: tan(5 * X) / 5,
    48: asinh(X),
    50: R(3, 10) * cos(X) * cosh(3 * X) + sin(X) * sinh(3 * X) / 10,
    51: -exp(-(X**2)) * (X**2 + 1) / 2,
    52: X * (cos(ln(X)) + sin(ln(X))) / 2,
    53: exp(sin(X**3)) / 3,
    54: X * asinh(X) - sqrt(1 + X**2),
    55: 9
    / (9 + ln(2) ** 2)
    * (-(2**X) * cos(3 * X) / 3 + ln(2) / 9 * 2**X * sin(3 * X)),
    56: X * ln(X) ** 2 - 2 * X * ln(X) + 2 * X,
}
"""Numero : la reponse imprimee, pages 64 a 67. Les 45 et 49 sont des sommes."""
SEUIL = 1e-12
RESOLUES = sorted(SOLUTIONS)


def _derive_en(primitive: sp.Expr, numero: int) -> float:
    return gestes.ecart(sp.diff(primitive, X), enonces.EXERCICE_UN[numero])


def test_les_cinquante_six_enonces_sont_la() -> None:
    assert sorted(enonces.EXERCICE_UN) == list(range(1, 57))
    assert sorted(un.resolutions()) == RESOLUES
    assert set(range(1, 57)) - set(RESOLUES) == {45, 49}


@pytest.mark.parametrize("numero", RESOLUES)
def test_la_reponse_du_syllabus_derive_en_l_integrande(numero: int) -> None:
    assert _derive_en(SOLUTIONS[numero], numero) < SEUIL


@pytest.mark.parametrize("numero", RESOLUES)
def test_chaque_primitive_derive_en_son_integrande(numero: int) -> None:
    resolution = un.resolutions()[numero]
    assert gestes.ecart(resolution.integrande, enonces.EXERCICE_UN[numero]) < SEUIL
    assert _derive_en(resolution.primitive, numero) < SEUIL
    assert resolution.methode
    assert resolution.etapes


@pytest.mark.parametrize("ordre", [1, 2, 3, 4])
def test_exercice_45_la_recurrence_des_puissances_du_logarithme(ordre: int) -> None:
    """Le syllabus : `x (ln^n x - n ln^(n-1) x + ... + (-1)^n n!)`."""
    imprimee = X * sum(
        (-1) ** k * sp.ff(ordre, k) * ln(X) ** (ordre - k) for k in range(ordre + 1)
    )
    calculee = un.puissance_du_logarithme(ordre)
    assert sp.expand(calculee - imprimee) == 0
    assert sp.simplify(sp.diff(calculee, X) - ln(X) ** ordre) == 0


@pytest.mark.parametrize("ordre", [1, 2, 3, 4])
def test_exercice_49_la_recurrence_de_l_exponentielle(ordre: int) -> None:
    """Le syllabus : `-e^-x (x^n + n x^(n-1) + ... + n!)`."""
    imprimee = -exp(-X) * sum(
        sp.ff(ordre, k) * X ** (ordre - k) for k in range(ordre + 1)
    )
    calculee = un.puissance_fois_exponentielle(ordre)
    assert sp.expand(calculee - imprimee) == 0
    assert sp.simplify(sp.diff(calculee, X) - X**ordre * exp(-X)) == 0


def test_les_deux_relations_de_recurrence() -> None:
    """`I_n = x ln^n x - n I_(n-1)` et `I_n = -x^n e^-x + n I_(n-1)`."""
    crochet, facteur = un.relation_45()
    assert crochet == X * ln(X) ** N
    assert facteur == -N
    crochet, facteur = un.relation_49()
    assert crochet == -(X**N) * exp(-X)
    assert facteur == N


def test_exercice_40_par_l_exponentielle_complexe() -> None:
    resolution = un.resolutions()[40]
    assert "complexe" in resolution.methode
    assert not resolution.primitive.has(sp.I)


def test_une_reecriture_fausse_est_refusee() -> None:
    with pytest.raises(ValueError, match="ne vaut pas l'integrande"):
        gestes.terme_a_terme(sin(X) ** 2, (1 + cos(2 * X)) / 2, "linearisation")


def test_un_changement_de_variable_faux_est_refuse() -> None:
    with pytest.raises(ValueError, match="ne redonne pas l'integrande"):
        gestes.changement(X * exp(X**2), exp(gestes.U), X**2)
    with pytest.raises(ValueError, match="ne redonne pas l'integrande"):
        gestes.changement_inverse(exp(sqrt(X)), gestes.T**2, sqrt(X), exp(gestes.T))


def test_une_primitive_immediate_fausse_est_refusee() -> None:
    with pytest.raises(ValueError, match="n'est pas une primitive"):
        gestes.immediate(1 / sin(X) ** 2, sp.cot(X))


def test_l_ecart_se_mesure_meme_avec_des_parametres() -> None:
    assert gestes.ecart(A * X + B, B + A * X) < SEUIL
    assert gestes.ecart(A * X, B * X) > SEUIL


def test_un_changement_montre_sa_variable_puis_l_integrale_transformee() -> None:
    resolution = un.resolutions()[17]
    assert resolution.etapes[0] == 5 * X**2 + 18
    assert resolution.etapes[1] == 1 / (10 * gestes.U)
    resolution = un.resolutions()[29]
    assert resolution.etapes[0] == gestes.T**2


def test_une_integration_par_parties_montre_l_integrale_qui_reste_simplifiee() -> None:
    """Au 54, `v du` se simplifie en `x / racine(1 + x^2)` : c'est ce qu'on montre."""
    (etape,) = un.resolutions()[54].etapes
    (restante,) = etape.atoms(sp.Integral)
    assert sp.simplify(restante.function - X / sqrt(1 + X**2)) == 0
    assert restante.function == X / sqrt(X**2 + 1)


def test_des_parties_en_cascade_abaissent_la_puissance_du_logarithme() -> None:
    """Au 46 : `ln^3`, puis `ln^2`, puis `ln`, toujours contre `x^4`."""
    etapes = un.resolutions()[46].etapes
    assert len(etapes) == 3
    restantes = [next(iter(e.atoms(sp.Integral))).function for e in etapes]
    assert sp.simplify(restantes[0] - R(3, 5) * X**4 * ln(X) ** 2) == 0
    assert sp.simplify(restantes[2] - X**4 / 5) == 0
    assert len(un.resolutions()[56].etapes) == 2


@pytest.mark.parametrize("numero", RESOLUES)
def test_la_presentation_ne_change_pas_la_primitive(numero: int) -> None:
    brute = un.resolutions()[numero].primitive
    presentee = presentation.presenter(numero, brute)
    assert gestes.ecart(sp.diff(presentee, X), sp.diff(brute, X)) < SEUIL
    assert not presentee.has(sp.pi)


def test_la_presentation_range_ce_que_la_simplification_melange() -> None:
    assert presentation.presenter(12, un.resolutions()[12].primitive) == tan(X) - X
    quatre = presentation.presenter(4, un.resolutions()[4].primitive)
    assert quatre.is_Mul
    vingt_six = presentation.presenter(26, un.resolutions()[26].primitive)
    assert vingt_six.has(sin(ln(X)))
    assert vingt_six.has(cos(ln(X)))


def test_le_40_se_lit_un_polynome_par_fonction_trigonometrique() -> None:
    quarante = presentation.presenter(40, un.resolutions()[40].primitive)
    assert quarante.is_Mul
    assert quarante.atoms(sp.sin) == {sin(3 * X)}
    assert quarante.atoms(sp.cos) == {cos(3 * X)}


def test_le_jeu_de_macros_de_l_exercice_un() -> None:
    macros = macros_de_l_exercice_un()
    assert all(nom.startswith("PrimUn") and nom.isalpha() for nom in macros)
    assert all(valeur.strip() for valeur in macros.values())
    assert macros["PrimUnDixSeptVariable"] == "5 x^{2} + 18"
    assert macros["PrimUnDixSeptDifferentielle"] == "10 x"
    assert macros["PrimUnDixSeptTransformee"] == r"\frac{1}{10 u}"
    assert macros["PrimUnVingtNeufDifferentielle"] == "2 t"
    assert macros["PrimUnDouzePrimitive"] == r"- x + \tan{\left(x \right)}"
    assert "PrimUnVingtDeuxEtapeD" in macros
    assert "PrimUnVingtDeuxEtapeE" not in macros
    assert "PrimUnQuinzeEtapeA" not in macros
    assert r"\ln{\left(x \right)}^{3}" in macros["PrimUnQuaranteCinqTrois"]
    assert "recueil-primitives-1" in JEUX_DE_MACROS


def test_l_ecart_est_relatif_a_la_taille_des_valeurs() -> None:
    """Au 13, les valeurs depassent `10^14` : un ecart absolu n'y dirait rien."""
    grand = 10**20 * X
    assert gestes.ecart(grand + R(1, 1000), grand) < SEUIL


def test_la_primitive_de_un_sur_u_porte_une_valeur_absolue() -> None:
    assert gestes.valeur_absolue(ln(X)) == ln(sp.Abs(X))
    assert un.resolutions()[6].primitive == -ln(sp.Abs(cos(X)))
