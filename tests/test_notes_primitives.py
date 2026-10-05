"""Documents de travail sur les primitives (N. Vast, 2005-2006) et notes manuscrites.

Source : `Mathematique/SYLLABUS/Syllabus_Notes_de_cours_2011.pdf`, pages 43 a
56 et 67 a 69 du fichier (un scan, lu en image). Les exemples detailles des
trois documents : primitives quasi immediates, par parties, fractions
rationnelles a racines simples puis multiples, fonctions trigonometriques
selon leur parite, fonctions irrationnelles (cas 1 a 8 du formulaire).

Chaque primitive annoncee est controlee par derivation, en des points de
son domaine quand des radicaux empechent une simplification formelle.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.integration.fractions import FractionRationnelle
from polytech.maths.domain.integration.hermite import (
    contribution_d_une_racine,
    decomposition_d_hermite,
    inverse_d_une_puissance,
)
from polytech.maths.domain.integration.irrationnelles import (
    indice_commun,
    par_changement_inverse,
)
from polytech.maths.domain.integration.regles_de_parite import (
    Changement,
    changement_conseille,
)
from polytech.maths.domain.integration.substitution import Substitution
from polytech.maths.domain.symboles import VARIABLE as X

R = sp.Rational
T = sp.Symbol("t", positive=True)
sin, cos, tan, sqrt, ln = sp.sin, sp.cos, sp.tan, sp.sqrt, sp.log


def _est_primitive(
    primitive: sp.Expr, integrande: sp.Expr, points: tuple[object, ...]
) -> bool:
    """Vrai si la derivee vaut l'integrande en chacun des points donnes."""
    ecart = sp.diff(primitive, X) - integrande
    return all(abs(sp.N(ecart.subs(X, point), 30)) < R(1, 10**20) for point in points)


def test_la_racine_d_une_fonction_affine() -> None:
    """Premiere partie, ex. 1.3 : `int racine(a + bx) = 2 (a + bx)^(3/2) / (3b)`."""
    a, b = sp.symbols("a b", positive=True)
    primitive = 2 * (a + b * X) ** R(3, 2) / (3 * b)
    assert sp.simplify(sp.diff(primitive, X) - sqrt(a + b * X)) == 0
    obtenue = Substitution(sqrt(a + b * X), a + b * X).resultat()
    assert sp.simplify(sp.diff(obtenue, X) - sqrt(a + b * X)) == 0


def test_les_parties_repetees_de_l_exercice_42() -> None:
    """`int (x+1)^3 ch x = (x+1)^3 sh x - 3(x+1)^2 ch x + 6(x+1) sh x - 6 ch x`."""
    y = X + 1
    primitive = (
        y**3 * sp.sinh(X) - 3 * y**2 * sp.cosh(X) + 6 * y * sp.sinh(X) - 6 * sp.cosh(X)
    )
    assert sp.simplify(sp.diff(primitive, X) - y**3 * sp.cosh(X)) == 0


def test_les_parties_qui_reviennent_au_depart_exercice_55() -> None:
    """`int 2^x sin 3x = 9/(9 + ln^2 2) (-2^x cos 3x / 3 + ln 2 . 2^x sin 3x / 9)`."""
    facteur = 9 / (9 + ln(2) ** 2)
    primitive = facteur * (-(2**X) * cos(3 * X) / 3 + ln(2) * 2**X * sin(3 * X) / 9)
    assert sp.simplify(sp.diff(primitive, X) - 2**X * sin(3 * X)) == 0


def test_les_exemples_manuscrits_de_changement_de_variable() -> None:
    """`tan x`, `e^x/(1 + e^(2x))`, `x/(2x^2 + 3)`, `sin^3 x cos x`, `3/(x^2+x+1)`."""
    couples = (
        (-ln(cos(X)), tan(X)),
        (sp.atan(sp.exp(X)), sp.exp(X) / (1 + sp.exp(2 * X))),
        (ln(2 * X**2 + 3) / 4, X / (2 * X**2 + 3)),
        (sin(X) ** 4 / 4, sin(X) ** 3 * cos(X)),
        (2 * sqrt(3) * sp.atan(2 * (X + R(1, 2)) / sqrt(3)), 3 / (X**2 + X + 1)),
    )
    for primitive, integrande in couples:
        assert sp.simplify(sp.diff(primitive, X) - integrande) == 0


def test_les_exemples_manuscrits_par_parties() -> None:
    """`x 2^x`, `ln x`, `arctan x`."""
    couples = (
        (2**X * (X - 1 / ln(2)) / ln(2), X * 2**X),
        (X * (ln(X) - 1), ln(X)),
        (X * sp.atan(X) - ln(1 + X**2) / 2, sp.atan(X)),
    )
    for primitive, integrande in couples:
        assert sp.simplify(sp.diff(primitive, X) - integrande) == 0


def test_la_division_avant_la_decomposition() -> None:
    """Manuscrit, page 4 : `(3x^3 + 2x + 1)/(x^2 - x - 1)`, `K = 3x + 3`, `R = 8x+4`."""
    fraction = FractionRationnelle(3 * X**3 + 2 * X + 1, X**2 - X - 1)
    assert not fraction.est_propre()
    assert fraction.partie_entiere() == 3 * X + 3
    reste = sp.expand(3 * X**3 + 2 * X + 1 - (3 * X + 3) * (X**2 - X - 1))
    assert reste == 8 * X + 4


def test_la_decomposition_de_l_exemple_manuscrit() -> None:
    """`(3x^2 + 1)/(x^3 + 1) = (4/3)/(x + 1) + (5x/3 - 1/3)/(x^2 - x + 1)`."""
    termes = FractionRationnelle(3 * X**2 + 1, X**3 + 1).termes()
    attendue = R(4, 3) / (X + 1) + (R(5, 3) * X - R(1, 3)) / (X**2 - X + 1)
    assert sp.simplify(sum(termes) - attendue) == 0
    assert (3 * X**2 + 1).subs(X, -1) / (X**2 - X + 1).subs(X, -1) == R(4, 3)


def test_les_racines_simples_par_les_coefficients() -> None:
    """Deuxieme partie, ex. 2f : `x/(x^4 - 1)`, `A = B = 1/4`, `C = -1/2`, `D = 0`."""
    fraction = FractionRationnelle(X, X**4 - 1)
    attendue = R(1, 4) / (X - 1) + R(1, 4) / (X + 1) - R(1, 2) * X / (X**2 + 1)
    assert sp.simplify(sum(fraction.termes()) - attendue) == 0
    primitive = ln(X - 1) / 4 + ln(X + 1) / 4 - ln(X**2 + 1) / 4
    assert sp.simplify(sp.diff(primitive, X) - X / (X**4 - 1)) == 0


def test_les_racines_simples_par_leurs_contributions() -> None:
    """Seconde methode : `G(1) = 1/4`, `G(-1) = 1/4`, `G(i) = -1/4`."""
    fraction = X / (X**4 - 1)
    assert contribution_d_une_racine(fraction, 1) == R(1, 4)
    assert contribution_d_une_racine(fraction, -1) == R(1, 4)
    assert contribution_d_une_racine(fraction, sp.I) == -R(1, 4)
    assert contribution_d_une_racine(fraction, -sp.I) == -R(1, 4)


def test_une_valeur_qui_n_est_pas_racine_simple_n_a_pas_de_contribution() -> None:
    """Une racine double, puis un point qui n'est pas racine du tout."""
    with pytest.raises(ValueError, match="pas une racine simple"):
        contribution_d_une_racine(1 / (X - 1) ** 2, 1)
    with pytest.raises(ValueError, match="pas une racine simple"):
        contribution_d_une_racine(X / (X**4 - 1), 2)


def test_les_racines_multiples_par_la_variante_d_hermite() -> None:
    """Ex. 2g : `(x+1)/(x^2+4x+5)^2 = ((Ax + B)/(x^2+4x+5))' + (Cx + D)/(x^2+4x+5)`.

    `A = -1/2 = D`, `B = -3/2`, `C = 0`.
    """
    rationnelle, denominateur, reste, simple = decomposition_d_hermite(
        X + 1, (X**2 + 4 * X + 5) ** 2
    )
    assert sp.expand(rationnelle - (-X / 2 - R(3, 2))) == 0
    assert sp.expand(denominateur - (X**2 + 4 * X + 5)) == 0
    assert reste == -R(1, 2)
    assert sp.expand(simple - (X**2 + 4 * X + 5)) == 0


def test_la_primitive_de_l_exemple_a_racines_multiples() -> None:
    """`-(x + 3) / (2 (x^2 + 4x + 5)) - arctan(x + 2) / 2`."""
    primitive = -(X + 3) / (2 * (X**2 + 4 * X + 5)) - sp.atan(X + 2) / 2
    assert sp.simplify(sp.diff(primitive, X) - (X + 1) / (X**2 + 4 * X + 5) ** 2) == 0


def test_une_fraction_sans_racine_multiple_n_a_pas_de_partie_rationnelle() -> None:
    """`1/(x^2 + 1)` : `Q_1 = 1`, rien a mettre en derivee."""
    rationnelle, denominateur, reste, simple = decomposition_d_hermite(1, X**2 + 1)
    assert (rationnelle, denominateur, reste) == (0, 1, 1)
    assert sp.expand(simple - (X**2 + 1)) == 0


def test_la_recurrence_des_inverses_de_puissances() -> None:
    """Recueil : `I_n = (2n-3)/(2n-2) I_(n-1) + x / ((2n-2)(x^2+1)^(n-1))`."""
    assert inverse_d_une_puissance(1) == sp.atan(X)
    deuxieme = sp.atan(X) / 2 + X / (2 * (1 + X**2))
    assert sp.simplify(inverse_d_une_puissance(2) - deuxieme) == 0
    troisieme = R(3, 4) * deuxieme + X / (4 * (1 + X**2) ** 2)
    assert sp.simplify(inverse_d_une_puissance(3) - troisieme) == 0
    for n in (2, 3, 4):
        derivee = sp.diff(inverse_d_une_puissance(n), X)
        assert sp.simplify(derivee - 1 / (X**2 + 1) ** n) == 0


def test_la_parite_des_exemples_de_la_troisieme_partie() -> None:
    """Impaire en sinus : `u = cos x`. Paire en sinus et en cosinus : `t = tan x`."""
    impaire = sin(X) ** 3 / (cos(X) * (2 - sin(X) ** 2))
    assert changement_conseille(impaire) is Changement.COSINUS
    paire = 1 / (1 + cos(X) ** 2)
    assert changement_conseille(paire) is Changement.TANGENTE
    ni_l_une_ni_l_autre = 1 / (2 + 5 * cos(X))
    assert changement_conseille(ni_l_une_ni_l_autre) is Changement.ARC_MOITIE
    assert changement_conseille(cos(X) ** 3 / (1 + sin(X) ** 2)) is Changement.SINUS


def test_une_fonction_rationnelle_en_tangente() -> None:
    """Troisieme partie, 4 : `(1 + tan x)/(1 - tan x)` appelle `t = tan x`."""
    assert changement_conseille((1 + tan(X)) / (1 - tan(X))) is Changement.TANGENTE


def test_la_fonction_impaire_en_sinus_de_l_exercice_7d() -> None:
    """`int sin x cos^2 x / (1 + sin^2 x)`, par `u = cos x`."""
    integrande = sin(X) * cos(X) ** 2 / (1 + sin(X) ** 2)
    assert changement_conseille(integrande) is Changement.COSINUS
    racine = sqrt(2)
    primitive = cos(X) + ln(racine - cos(X)) / racine - ln(racine + cos(X)) / racine
    assert sp.simplify(sp.diff(primitive, X) - integrande) == 0


def test_la_fonction_paire_de_la_troisieme_partie() -> None:
    """`int 1/(1 + cos^2 x) = arctan(tan x / racine(2)) / racine(2)`, `t = tan x`."""
    integrande = 1 / (1 + cos(X) ** 2)
    primitive = sp.atan(tan(X) / sqrt(2)) / sqrt(2)
    assert _est_primitive(primitive, integrande, (R(1, 3), 1, R(5, 4)))
    en_t = sp.simplify((1 / (1 + 1 / (1 + T**2))) / (1 + T**2))
    assert sp.simplify(en_t - 1 / (2 + T**2)) == 0


def test_l_indice_commun_des_radicaux() -> None:
    """Cas 2 : pour `racine(x)` et `x^(1/3)`, « on prend `k = 6` », `t^6 = x`."""
    assert indice_commun(2, 3) == 6
    assert indice_commun(2, 4) == 4
    assert indice_commun(3) == 3


def test_le_cas_1_la_racine_cubique_d_une_homographie() -> None:
    """Ex. 9b : `((x+1)/(x-1))^(1/3)`, par `t^3 = (x+1)/(x-1)`."""
    integrande = ((X + 1) / (X - 1)) ** R(1, 3)
    retour = ((X + 1) / (X - 1)) ** R(1, 3)
    primitive = par_changement_inverse(integrande, T, (T**3 + 1) / (T**3 - 1), retour)
    assert _est_primitive(primitive, integrande, (2, 3, R(7, 2)))


def test_le_cas_2_deux_radicaux_d_indices_differents() -> None:
    """Ex. 9d : `(racine(x) - 1)/(x^(1/3) + 1)`, par `t^6 = x`."""
    integrande = (sqrt(X) - 1) / (X ** R(1, 3) + 1)
    primitive = par_changement_inverse(integrande, T, T**6, X ** R(1, 6))
    assert _est_primitive(primitive, integrande, (1, 2, 5))
    sixieme = X ** R(1, 6)
    annoncee = 6 * (
        sixieme**7 / 7
        - sixieme**5 / 5
        - sixieme**4 / 4
        + sixieme**3 / 3
        + sixieme**2 / 2
        - sixieme
        - ln(sixieme**2 + 1) / 2
        + sp.atan(sixieme)
    )
    assert _est_primitive(annoncee, integrande, (1, 2, 5))


def test_le_cas_3_la_racine_d_un_trinome() -> None:
    """`int racine(5x^2 - 4x + 1)`, ramene a `racine(u^2 + 1)` avec `u = 5(x - 2/5)`."""
    u = 5 * (X - R(2, 5))
    primitive = sqrt(5) / 25 * (u / 2 * sqrt(u**2 + 1) + ln(u + sqrt(u**2 + 1)) / 2)
    assert _est_primitive(primitive, sqrt(5 * X**2 - 4 * X + 1), (0, 1, R(-3, 2)))


def test_le_cas_4_un_numerateur_du_premier_degre_sur_la_racine() -> None:
    """`int (x+3)/racine(5 - 4x - x^2) = -racine(5 - 4x - x^2) + arcsin((x+2)/3)`."""
    radical = sqrt(5 - 4 * X - X**2)
    primitive = -radical + sp.asin((X + 2) / 3)
    assert _est_primitive(primitive, (X + 3) / radical, (0, -2, R(1, 2)))


def test_le_cas_5_la_racine_d_un_trinome_a_racines_reelles() -> None:
    """Ex. : `(x - 1)/(x racine(-x^2 + 4x - 3))`, par `t^2 = (3 - x)/(x - 1)`."""
    integrande = (X - 1) / (X * sqrt(-(X**2) + 4 * X - 3))
    primitive = -2 * sp.atan(sqrt((3 - X) / (X - 1))) + 2 / sqrt(3) * sp.atan(
        sqrt((3 - X) / (3 * (X - 1)))
    )
    assert _est_primitive(primitive, integrande, (2, R(3, 2), R(5, 2)))


def test_le_cas_6_par_le_changement_d_euler() -> None:
    """Ex. 11c : `x^4/racine(x^2 + 1)`, par `t = racine(x^2 + 1) - x`."""
    integrande = X**4 / sqrt(X**2 + 1)
    retour = sqrt(X**2 + 1) - X
    primitive = par_changement_inverse(integrande, T, (1 - T**2) / (2 * T), retour)
    assert _est_primitive(primitive, integrande, (0, 1, 2))
    t = retour
    annoncee = -R(1, 16) * (
        -1 / (4 * t**4) + t**4 / 4 + 6 * ln(t) + 2 / t**2 - 2 * t**2
    )
    assert _est_primitive(annoncee, integrande, (0, 1, 2))


def test_le_cas_8_et_le_dernier_exercice() -> None:
    """`int 1/(x racine(x^2 - 4)) = arccos(2/x)/2` ; puis le dernier exercice."""
    assert _est_primitive(sp.acos(2 / X) / 2, 1 / (X * sqrt(X**2 - 4)), (3, 5, 10))
    radical = sqrt(4 * X**2 - 2 * X + 1)
    primitive = radical / 2 + R(3, 4) * sp.asinh(4 / sqrt(3) * (X - R(1, 4)))
    assert _est_primitive(primitive, (2 * X + 1) / radical, (0, 1, -2))


def test_un_changement_qui_laisse_un_radical_est_refuse() -> None:
    """`t = x` ne fait rien disparaitre de `racine(x^2 + 1)`."""
    with pytest.raises(ValueError, match="ne rend pas l'integrande rationnel"):
        par_changement_inverse(sqrt(X**2 + 1), T, T, X)
