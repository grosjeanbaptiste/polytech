"""Le recueil de primitives du syllabus de 2011, exercices 2 a 11.

Source : `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf`, chapitre
« Primitives », pages 61 a 63 du PDF ; l'oracle est « Solutions du
chapitre XII », pages 67 a 72. Le document est un scan : enonces et
reponses ont ete lus sur les pages.

Deux controles, comme pour l'exercice 1 : la derivee de chaque primitive
rend l'integrande, et celle de la reponse du syllabus aussi. Deux reponses
du syllabus n'y resistent pas ; l'agrandissement du scan confirme que ce
n'est pas une faute de lecture, et elles sont consignees a part.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.application import recueil_primitives_enonces as enonces
from polytech.maths.application import recueil_primitives_gestes as gestes
from polytech.maths.application import recueil_primitives_presentation as presentation
from polytech.maths.application import recueil_primitives_recurrence as recurrence
from polytech.maths.application import recueil_primitives_sommes as sommes
from polytech.maths.application import recueil_primitives_suite as suite
from polytech.maths.application.registre import JEUX_DE_MACROS

x, a, b, p = enonces.X, enonces.A, enonces.B, enonces.P
q, w, h = enonces.Q, enonces.W, enonces.H
R = sp.Rational
ln, sin, cos, tan, exp, sqrt = sp.log, sp.sin, sp.cos, sp.tan, sp.exp, sp.sqrt
atan, asin, sinh, cosh = sp.atan, sp.asin, sp.sinh, sp.cosh
atanh, acoth, A, asinh = sp.atanh, sp.acoth, sp.Abs, sp.asinh
u36 = 2 / sqrt(3) * (x + R(1, 2))
T = x**2 + x + 1
s = sp.cbrt((x + 1) / (x - 1))
z = sqrt(x**2 + 1) - x
rr = sqrt((2 - x) / (x + 1))
r2 = sqrt((2 - x) / (2 + x))
L = ln(100)

SOLUTIONS = {
    "2a": -ln(A(x)) / 4 + ln(A(x - 2)) / 8 + ln(A(x + 2)) / 8,
    "2b": ln(A(x)) / 4 - ln(x**2 + 4) / 8,
    "2c": ln(A(x - 1)) / 3 - ln(A(T)) / 6 + sqrt(3) / 3 * atan(u36),
    "2d": x - 1 / (3 * x) - R(7, 9) * ln(A(x)) - R(20, 9) * ln(A(x + 3)),
    "2e": ln(A(x - 1)) / 4 - ln(A(x + 1)) / 4 - atan(x) / 2,
    "2f": ln(A(x - 1)) / 4 + ln(A(x + 1)) / 4 - ln(x**2 + 1) / 4,
    "2g": -R(1, 2) * (x + 3) / (x**2 + 4 * x + 5) - atan(x + 2) / 2,
    "2h": R(1, 10) * (2 * x + 1) / (x**2 + 1)
    + ln(A(x + 2)) / 25
    - ln(1 + x**2) / 50
    + R(7, 25) * atan(x),
    "3b1": atan(x),
    "3b2": atan(x) / 2 + x / (2 * (1 + x**2)),
    "3b3": R(3, 4) * (atan(x) / 2 + x / (2 * (1 + x**2))) + x / (4 * (1 + x**2) ** 2),
    "3c1": 2 * sqrt(3) / 3 * atan(u36),
    "3c2": 8 * sqrt(3) / 9 * (atan(u36) / 2 + R(3, 8) * u36 / T),
    "4a": ln(exp(x) / (1 + exp(x))),
    "4b": (-3 * exp(2 * x) - R(9, 2) * exp(x) - 1) / (exp(x) * (1 + exp(x)) ** 2)
    - 3 * x
    + 3 * ln(1 + exp(x)),
    "4c": atan(2**x) / ln(2),
    "4d": R(1, 3)
    * (-3 * x / 2 + ln(A(exp(3 * x) - 1)) / 3 + ln(A(exp(3 * x) + 2)) / 6),
    "5a": x**2 / 2,
    "5b": -(100 ** (-4 * x))
    / (4 * L)
    * (x**3 + 3 * x**2 / (4 * L) + 3 * x / (8 * L**2) + 3 / (32 * L**3)),
    "5c": -cos(3 * x) / 3 * (x**3 + 4 - R(2, 3) * x)
    + sin(3 * x) / 3 * (x**2 - R(2, 9)),
    "5d": (sin(x) * cosh(x) - cos(x) * sinh(x)) / 2,
    "5e": -atan(x) / x + ln(A(x)) - ln(1 + x**2) / 2,
    "6a": -ln(A(2 - 7 * x**2)) / 14,
    "6b": (x**2 + 6) ** R(3, 2),
    "6c": sin(x) ** 3 / 3 - sin(x) ** 5 / 5,
    "6d": -ln(A(x)) + ln(A(x - 1)) + ln(A(x + 1)),
    "6e": ln(A((x - 2 - sqrt(2)) / (x - 2 + sqrt(2)))) / (2 * sqrt(2)),
    "6f": ln(A((rr - sqrt(2)) / (rr + sqrt(2)))) / sqrt(2),
    "6g": x / (3 * sqrt(3 + x**2)),
    "6h": ln(A(x - 1)) / 4 - ln(A(x + 1)) / 4 + atan(x) / 2,
    "7a": x**4 / 4 * atan(x) - R(1, 4) * (x**3 / 3 - x + atan(x)),
    "7b": -x * exp(x) / (1 + x) + exp(x),
    "7c": 1 / (2 * cos(x) ** 2) + ln(A(cos(x))),
    "7d": cos(x)
    + ln(A(sqrt(2) - cos(x))) / sqrt(2)
    - ln(A(sqrt(2) + cos(x))) / sqrt(2),
    "7e": (ln(A(1 - sqrt(3) * r2)) - ln(A(1 + sqrt(3) * r2))) / sqrt(3),
    "8a": -ln(A((sqrt(7) - sqrt(3) * tan(x / 2)) / (sqrt(7) + sqrt(3) * tan(x / 2))))
    / sqrt(21),
    "8b": atanh(sin(x)),
    "8c": b / (a**2 + b**2) * ln(A(a + b * tan(x)))
    - b / (2 * (a**2 + b**2)) * ln(tan(x) ** 2 + 1)
    + a * x / (a**2 + b**2),
    "8d": R(1, 8) * (x / 2 - sin(4 * x) / 8 - sin(2 * x) ** 3 / 6),
    "8e": R(1, 2) * (sin((a - b) * x) / (a - b) + sin((a + b) * x) / (a + b)),
    "8f": -(ln(A(tan(x / 2) - 1 - sqrt(2))) - ln(A(tan(x / 2) - 1 + sqrt(2))))
    / sqrt(2),
    "8g": -1 / sin(x) + atanh(sin(x)),
    "8h": -ln(A(1 - tan(x))) + ln(1 + tan(x) ** 2) / 2,
    "8i": R(1, 8)
    * (R(5, 2) * x + 2 * sin(2 * x) + R(3, 8) * sin(4 * x) - sin(2 * x) ** 3 / 6),
    "8j": -sin(h) * x / 2 - cos(2 * w * x + h) / (4 * w),
    "8k": -2 / (tan(x / 2) - 1) - x,
    "8l": -sin(x) + 2 * atan(sin(x)),
    "8m": ln((sqrt(3) + tan(x)) / (sqrt(3) - tan(x))) / (8 * sqrt(3)) + x / 4,
    "8n": ln(A(1 + 5 * tan(x / 2))) / 5,
    "8o": cos(x) ** 3 / 3 - cos(x),
    "8p": R(1, 4) * (-cos(2 * x) / 2 - cos(4 * x) / 4 + cos(6 * x) / 6),
    "9a": (sqrt(3 + 4 * x) - ln(A(1 + sqrt(3 + 4 * x)))) / 2,
    "9b": 2 * s / (s**3 - 1)
    - R(2, 3) * ln(A(s - 1))
    + ln(A(s**2 + s + 1)) / 3
    + 2 / sqrt(3) * atan((2 * s + 1) / sqrt(3)),
    "9c": 6 * (sqrt(x) / 3 - sp.cbrt(x) / 2 + x ** R(1, 6) - ln(x ** R(1, 6) + 1)),
    "9d": 6
    * (
        x ** R(7, 6) / 7
        - x ** R(5, 6) / 5
        - x ** R(4, 6) / 4
        + x ** R(3, 6) / 3
        + x ** R(2, 6) / 2
        - x ** R(1, 6)
        - ln(x ** R(1, 3) + 1) / 2
        + atan(x ** R(1, 6))
    ),
    "10aPlus": x / 2 * sqrt(x**2 + p**2)
    + p**2 / 2 * ln(A(x / p + sqrt(x**2 + p**2) / p)),
    "10aMoins": x / 2 * sqrt(x**2 - p**2)
    - p**2 / 2 * ln(A(x / p + sqrt(x**2 - p**2) / p)),
    "10b": p**2 / 2 * (asin(x / p) + x / p * sqrt(1 - x**2 / p**2)),
    "10c": -acoth(sqrt(x**2 + 1)),
    "10d": -sqrt(x**2 + 1) / x,
    "10f": ln(A(4 * x + 1 + sqrt((4 * x + 1) ** 2 - 1))) / sqrt(2),
    "11a": x + tan(x),
    "11b": -atan(x) / (2 * (1 + x) ** 2)
    - 1 / (4 * (1 + x))
    - ln(1 + x**2) / 8
    + ln(A(1 + x)) / 4,
    "11c": -R(1, 16)
    * (-1 / (4 * z**4) + z**4 / 4 + 6 * ln(A(z)) + 2 / z**2 - 2 * z**2),
    "11d": (3 * sqrt(x - 1) ** 3 + 5 * sqrt(x - 1)) / (4 * x**2)
    + R(3, 4) * atan(sqrt(x - 1)),
    "11e": sqrt(2) / 2 * atan(tan(2 * x) / sqrt(2)),
    "11f": x**2 / 2 * ln(1 + x**3)
    - R(3, 4) * x**2
    - ln(A(x + 1)) / 2
    + ln(x**2 - x + 1) / 4
    + sqrt(3) / 2 * atan(2 / sqrt(3) * (x - R(1, 2))),
    "11g": 3 * exp(x / 3),
    "11h": x * tan(x / 2),
    "11i": (x**2 + 1) ** 2 * cosh(x)
    - 4 * (x**3 + x) * sinh(x)
    + 4 * (3 * x**2 + 1) * cosh(x)
    - 24 * x * sinh(x)
    + 24 * cosh(x),
}
"""Cle de la primitive : la reponse imprimee, pages 67 a 72."""

IMPRIMEE_3C3 = (
    32
    * sqrt(3)
    / 27
    * (R(3, 8) * atan(u36) + R(3, 8) * u36 / T + R(9, 64) * u36 / T**2)
)
"""Page 68 : le second coefficient est imprime `3/8`. Il vaut `9/32`."""
IMPRIMEE_10E = asin(2 / (p - q) * (x - (p + q) / 2))
"""Page 71 : le denominateur est imprime `p - q`. Il vaut `q - p`."""
SEUIL = 1e-12
RECURRENCES = ("3b1", "3b2", "3b3", "3c1", "3c2", "3c3")
RESOLUES = sorted(set(SOLUTIONS) - set(RECURRENCES))


def _points(cle: str) -> tuple[sp.Expr, ...]:
    return enonces.POINTS_PARTICULIERS.get(cle, gestes.POINTS)


def _derive_en(primitive: sp.Expr, cle: str) -> float:
    integrande = enonces.EXERCICES_SUIVANTS[cle]
    return gestes.ecart(sp.diff(primitive, x), integrande, _points(cle))


def test_les_enonces_des_dix_exercices_sont_la() -> None:
    cles = set(enonces.EXERCICES_SUIVANTS)
    assert set(suite.resolutions()) == cles
    assert cles == (set(SOLUTIONS) - set(RECURRENCES)) | {"10e"}
    assert len(cles) == 66


@pytest.mark.parametrize("cle", RESOLUES)
def test_la_reponse_du_syllabus_derive_en_l_integrande(cle: str) -> None:
    assert _derive_en(SOLUTIONS[cle], cle) < SEUIL


@pytest.mark.parametrize("cle", [*RESOLUES, "10e"])
def test_chaque_primitive_derive_en_son_integrande(cle: str) -> None:
    resolution = suite.resolutions()[cle]
    integrande = enonces.EXERCICES_SUIVANTS[cle]
    assert gestes.ecart(resolution.integrande, integrande, _points(cle)) < SEUIL
    assert _derive_en(resolution.primitive, cle) < SEUIL
    assert resolution.methode
    assert resolution.etapes


def test_ecart_au_syllabus_le_signe_du_10_e() -> None:
    """`racine(x - p) racine(q - x)` demande `p < x < q` : l'arc sinus croit."""
    assert _derive_en(IMPRIMEE_10E, "10e") > 1
    assert _derive_en(-IMPRIMEE_10E, "10e") < SEUIL


def test_ecart_au_syllabus_le_coefficient_du_3_c() -> None:
    integrande = 1 / T**3
    imprimee = gestes.ecart(sp.diff(IMPRIMEE_3C3, x), integrande)
    assert imprimee > 1e-3
    corrigee = (
        32
        * sqrt(3)
        / 27
        * (R(3, 8) * atan(u36) + R(9, 32) * u36 / T + R(9, 64) * u36 / T**2)
    )
    assert gestes.ecart(sp.diff(corrigee, x), integrande) < SEUIL
    calculee = recurrence.sur_le_trinome(3)
    assert gestes.ecart(sp.diff(calculee, x), integrande) < SEUIL
    assert sp.simplify(calculee - corrigee) == 0


@pytest.mark.parametrize("ordre", [1, 2, 3])
def test_exercice_3_la_recurrence(ordre: int) -> None:
    calculee = recurrence.puissance(ordre)
    assert sp.simplify(sp.diff(calculee, x) - 1 / (x**2 + 1) ** ordre) == 0
    assert sp.simplify(calculee - SOLUTIONS[f"3b{ordre}"]) == 0


@pytest.mark.parametrize("ordre", [1, 2])
def test_exercice_3_le_trinome(ordre: int) -> None:
    calculee = recurrence.sur_le_trinome(ordre)
    assert sp.simplify(calculee - SOLUTIONS[f"3c{ordre}"]) == 0


def test_exercice_3_la_relation_se_verifie_en_derivant() -> None:
    """`I_n = (2n-3)/(2n-2) I_(n-1) + x / ((2n-2)(x^2+1)^(n-1))`."""
    facteur, terme = recurrence.relation()
    n = enonces.N
    assert facteur == (2 * n - 3) / (2 * n - 2)
    for ordre in (2, 3, 4):
        gauche = 1 / (x**2 + 1) ** ordre
        droite = facteur.subs(n, ordre) / (x**2 + 1) ** (ordre - 1) + sp.diff(
            terme.subs(n, ordre), x
        )
        assert sp.simplify(gauche - droite) == 0


def test_un_ordre_nul_est_refuse() -> None:
    with pytest.raises(ValueError, match="entier strictement positif"):
        recurrence.puissance(0)


def test_un_logarithme_ne_garde_pas_de_facteur_constant() -> None:
    """`ln |8x - 16|` et `ln |x - 2|` ne different que d'une constante."""
    logarithmes = suite.resolutions()["2a"].primitive.atoms(sp.log)
    assert {terme.args[0] for terme in logarithmes} == {A(x), A(x - 2), A(x + 2)}
    assert gestes.valeur_absolue(ln(8 * x - 16)) == ln(A(x - 2))
    assert gestes.valeur_absolue(ln(2 - 7 * x**2)) == ln(A(7 * x**2 - 2))


@pytest.mark.parametrize("cle", [*RESOLUES, "10e"])
def test_la_presentation_ne_change_pas_la_primitive(cle: str) -> None:
    brute = suite.resolutions()[cle].primitive
    presentee = presentation.presenter(cle, brute)
    ecart = gestes.ecart(sp.diff(presentee, x), sp.diff(brute, x), _points(cle))
    assert ecart < SEUIL
    assert not presentee.has(sp.Piecewise)


def test_la_presentation_range_les_ecritures_de_sympy() -> None:
    presenter, resolues = presentation.presenter, suite.resolutions()
    assert presenter("10d", resolues["10d"].primitive) == -sqrt(x**2 + 1) / x
    assert presenter("6g", resolues["6g"].primitive) == x / (3 * sqrt(x**2 + 3))
    assert not presenter("8m", resolues["8m"].primitive).has(atan(tan(x)))
    deux_g = presenter("2g", resolues["2g"].primitive)
    assert len(sp.Add.make_args(deux_g)) == 2
    dix = presenter("10aPlus", resolues["10aPlus"].primitive)
    assert dix == p**2 * asinh(x / p) / 2 + x * sqrt(p**2 + x**2) / 2


def test_le_5_b_garde_son_exposant_sous_un_nom() -> None:
    """`100^(-4x) = e^(-kx)`, `k = 4 ln 100` : trois parties, sans puissances de 100."""
    resolution = suite.resolutions()["5b"]
    assert resolution.integrande == x**3 * exp(-enonces.K * x)
    assert gestes.VALEURS[enonces.K] == 4 * ln(100)


def test_le_8_e_suppose_des_frequences_distinctes() -> None:
    assert not suite.resolutions()["8e"].primitive.has(sp.Piecewise)


def test_un_logarithme_multiplie_par_une_fonction_garde_son_argument() -> None:
    """`x^(p+1) ln(ax)` : ici le facteur `a` n'est pas une constante additive."""
    produit = x**2 * ln(a * x)
    assert gestes.valeur_absolue(produit) == x**2 * ln(A(a * x))
    assert gestes.valeur_absolue(3 * ln(a * x)) == 3 * ln(A(x))


def test_le_5_c_garde_l_angle_triple() -> None:
    cinq_c = presentation.presenter("5c", suite.resolutions()["5c"].primitive)
    assert cinq_c.atoms(sp.sin) == {sin(3 * x)}
    assert cinq_c.atoms(sp.cos) == {cos(3 * x)}


def test_exercice_3_les_coefficients_que_le_syllabus_ecrit() -> None:
    """Devant `arctg u`, `u / T`, `u / T^2` : le syllabus imprime `3/8, 3/8, 9/64`."""
    assert recurrence.coefficients_du_trinome(2) == (R(1, 2), R(3, 8))
    assert recurrence.coefficients_du_trinome(3) == (R(3, 8), R(9, 32), R(9, 64))
    c0, c1, c2 = recurrence.coefficients_du_trinome(3)
    forme = recurrence.facteur_du_trinome(3) * (
        c0 * atan(u36) + c1 * u36 / T + c2 * u36 / T**2
    )
    assert sp.simplify(forme - recurrence.sur_le_trinome(3)) == 0


JEUX = {
    "recueil-primitives-2": ("PrimDeux", 12),
    "recueil-primitives-3": ("PrimTrois", 18),
    "recueil-primitives-4": ("PrimQuatre", 16),
    "recueil-primitives-5": ("PrimCinq", 20),
}
"""Dossier : le prefixe de ses macros, et le nombre de ses primitives."""


@pytest.mark.parametrize("dossier", sorted(JEUX))
def test_chaque_dossier_a_son_jeu_de_macros(dossier: str) -> None:
    prefixe, nombre = JEUX[dossier]
    macros = JEUX_DE_MACROS[dossier]()
    assert all(nom.startswith(prefixe) and nom.isalpha() for nom in macros)
    assert all(valeur.strip() for valeur in macros.values())
    assert sum(nom.endswith("Integrande") for nom in macros) == nombre
    assert sum(nom.endswith("Primitive") for nom in macros) == nombre


def test_les_macros_de_la_recurrence_et_d_un_changement() -> None:
    macros = JEUX_DE_MACROS["recueil-primitives-2"]()
    assert macros["PrimDeuxTroisCoefficientsTrois"] == (
        r"\frac{3}{8},\; \frac{9}{32},\; \frac{9}{64}"
    )
    assert "PrimDeuxTroisTrinomeTrois" in macros
    assert macros["PrimDeuxQuatreCVariable"] == "2^{x}"
    assert macros["PrimDeuxDeuxAEtapeA"].count(r"\frac") == 3


def test_une_differentielle_qui_est_une_somme_s_ecrit_entre_parentheses() -> None:
    """`du = (tan^2 x + 1) dx`, et non `du = tan^2 x + 1 dx`."""
    macros = JEUX_DE_MACROS["recueil-primitives-4"]()
    assert macros["PrimQuatreHuitMDifferentielle"].startswith(r"\left(")
    assert macros["PrimQuatreHuitMDifferentielle"].endswith(r"\right)")
    assert macros["PrimQuatreHuitBDifferentielle"] == r"\cos{\left(x \right)}"


@pytest.mark.parametrize("ordre", [1, 2, 3])
def test_exercice_3_la_forme_que_le_syllabus_ecrit(ordre: int) -> None:
    """En `u` et `T` : c'est elle que le corrige affiche, sans fractions emboitees."""
    forme = recurrence.forme_du_trinome(ordre)
    assert forme.free_symbols <= {recurrence.U, recurrence.TRINOME}
    en_x = forme.subs({recurrence.U: recurrence.CENTRE, recurrence.TRINOME: T})
    assert sp.simplify(en_x - recurrence.sur_le_trinome(ordre)) == 0


def test_une_differentielle_rationnelle_est_reduite() -> None:
    """Au 9 b), `dx = -6 t^2 / (t^3 - 1)^2 dt`, d'une seule fraction."""
    macros = JEUX_DE_MACROS["recueil-primitives-5"]()
    assert macros["PrimCinqNeufBDifferentielle"].count(r"\frac") == 1


def test_un_morceau_mal_primitive_ou_une_somme_fausse_sont_refuses() -> None:
    faux = ((sp.Integer(1), x), (1 + tan(x) ** 2, sp.cot(x)))
    with pytest.raises(ValueError, match="n'est pas une primitive du terme"):
        sommes.par_morceaux(2 + tan(x) ** 2, faux, "decomposition")
    incomplet = ((sp.Integer(1), x),)
    with pytest.raises(ValueError, match="ne redonnent pas l'integrande"):
        sommes.par_morceaux(2 + tan(x) ** 2, incomplet, "decomposition")


def test_une_derivee_mal_reconnue_est_refusee() -> None:
    integrande = enonces.EXERCICES_SUIVANTS["11h"]
    juste = x / (2 * cos(x / 2) ** 2) + tan(x / 2)
    with pytest.raises(ValueError, match="ne vaut pas l'integrande"):
        sommes.reconnue(integrande, juste + 1, x * tan(x / 2), "derivee d'un produit")
    with pytest.raises(ValueError, match="n'est pas une primitive de"):
        sommes.reconnue(integrande, juste, x * tan(x), "derivee d'un produit")
