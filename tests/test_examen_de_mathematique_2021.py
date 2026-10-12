"""L'examen de mathematique de janvier 2021, refait par la bibliotheque.

Source : `Mathematique/EXAMENS/Exam_2021_Janvier_Enonce.pdf`, quatre pages,
quatre questions, trois options chacune : douze problemes. Le sujet n'a pas
de corrige au depot. Chaque reponse est donc etablie par la bibliotheque
puis recoupee par un chemin qui ne la suppose pas : une primitive est
derivee, une solution reportee dans son equation, une recurrence comparee a
l'integrale calculee directement.

Le systeme (6) est celui de janvier 2025, deja tenu par
`tests/test_algebre_discussion_par_le_rang.py`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.algebre.discussion import discuter
from polytech.maths.domain.algebre.discussion_par_le_rang import discuter_par_le_rang
from polytech.maths.domain.algebre.solutions import Nature
from polytech.maths.domain.algebre.systemes import SystemeLineaire
from polytech.maths.domain.differentielles.lineaire import EquationLineaire
from polytech.maths.domain.differentielles.premier_ordre import VARIABLE as T
from polytech.maths.domain.integration.fractions import FractionRationnelle
from polytech.maths.domain.integration.parties import IntegrationParParties
from polytech.maths.domain.symboles import CONSTANTE as C
from polytech.maths.domain.symboles import VARIABLE as X

A = sp.Symbol("a", real=True)
POSITIF = sp.Symbol("x", positive=True)


def _derive_en(primitive: sp.Expr, integrande: sp.Expr, variable: sp.Symbol) -> bool:
    return bool(sp.simplify(sp.diff(primitive, variable) - integrande) == 0)


# --- Question 1 : une primitive au choix ------------------------------------


def test_q1_i_par_l_arc_moitie_puis_par_parties() -> None:
    """`1 + cos x = 2 cos^2(x/2)` ; `u = x`, `dv = dx / (2 cos^2(x/2))`."""
    assert sp.simplify(1 + sp.cos(X) - 2 * sp.cos(X / 2) ** 2) == 0
    parties = IntegrationParParties(X, 1 / (2 * sp.cos(X / 2) ** 2))
    assert sp.simplify(parties.v() - sp.tan(X / 2)) == 0
    primitive = X * sp.tan(X / 2) + 2 * sp.log(sp.cos(X / 2))
    assert _derive_en(primitive, X / (1 + sp.cos(X)), X)


def test_q1_j_par_les_fractions_simples() -> None:
    """`(1 + 2x - 3x^2) / (x - x^3) = 1/x + 2/(x + 1)` : le facteur `1 - x` tombe."""
    fraction = FractionRationnelle(1 + 2 * X - 3 * X**2, X - X**3)
    assert sp.simplify(sum(fraction.termes()) - (1 / X + 2 / (X + 1))) == 0
    primitive = sp.log(POSITIF) + 2 * sp.log(POSITIF + 1)
    integrande = (1 + 2 * POSITIF - 3 * POSITIF**2) / (POSITIF - POSITIF**3)
    assert _derive_en(primitive, integrande, POSITIF)


def test_q1_k_par_le_changement_de_variable_logarithme() -> None:
    """`xi = ln x` ramene a `xi / (1 - xi^2)`, de primitive `-ln|1 - xi^2| / 2`."""
    xi = sp.Symbol("xi", real=True)
    assert _derive_en(-sp.log(1 - xi**2) / 2, xi / (1 - xi**2), xi)
    primitive = -sp.log(1 - sp.log(POSITIF) ** 2) / 2
    integrande = sp.log(POSITIF) / (POSITIF * (1 - sp.log(POSITIF) ** 2))
    assert _derive_en(primitive, integrande, POSITIF)


# --- Question 2 : un systeme a discuter -------------------------------------


def _systeme(lignes: list[list[sp.Expr]]) -> SystemeLineaire:
    complete = sp.Matrix(lignes)
    return SystemeLineaire(complete[:, :-1], complete[:, -1])


SYSTEME_4 = _systeme(
    [
        [A, A + 1, A - 1, 2 * A + 3],
        [1, 1 + A, 1 - A, 4 * A + 1],
        [A + 1, 2 * (A + 1), 0, 6 * A + 4],
    ]
)
SYSTEME_5 = _systeme([[A, A**2, A**3, 0], [1, 1, 1, 0], [A**3, A**2, A, 0]])


def test_q2_le_systeme_4_a_une_equation_de_trop() -> None:
    """La troisieme ligne est la somme des deux premieres, second membre compris."""
    complete = SYSTEME_4.matrice.row_join(SYSTEME_4.second_membre)
    assert sp.expand(complete.row(0) + complete.row(1) - complete.row(2)) == sp.zeros(
        1, 4
    )
    with pytest.raises(ValueError, match="determinant nul pour toute valeur de a"):
        discuter(SYSTEME_4, A)


def test_q2_le_systeme_4_est_indetermine_sauf_en_moins_un() -> None:
    discussion = discuter_par_le_rang(SYSTEME_4, A)
    assert discussion.generique is Nature.INDETERMINE
    assert discussion.particulieres[sp.Integer(-1)] is Nature.IMPOSSIBLE
    en_moins_un = discussion.en(sp.Integer(-1))
    complete = en_moins_un.matrice.row_join(en_moins_un.second_membre)
    assert complete.row(0) + complete.row(1) == sp.Matrix([[0, 0, 0, -2]])


def test_q2_le_systeme_5_n_a_que_la_solution_nulle_hors_de_trois_valeurs() -> None:
    """Homogene : jamais impossible. `det = -a^2 (a - 1)^3 (a + 1)`."""
    determinant = sp.factor(SYSTEME_5.matrice.det())
    assert sp.expand(determinant + A**2 * (A - 1) ** 3 * (A + 1)) == 0
    assert discuter(SYSTEME_5, A) == {
        sp.Integer(-1): Nature.INDETERMINE,
        sp.Integer(0): Nature.INDETERMINE,
        sp.Integer(1): Nature.INDETERMINE,
    }
    assert discuter_par_le_rang(SYSTEME_5, A).generique is Nature.UNIQUE
    assert SYSTEME_5.est_homogene()


# --- Question 3 : une equation lineaire du premier ordre --------------------


@pytest.mark.parametrize(
    ("p", "q", "depart", "valeur", "attendue"),
    [
        (-2 * T, T, 1, sp.E - sp.Rational(1, 2), sp.exp(T**2) - sp.Rational(1, 2)),
        (sp.tan(T), sp.sin(2 * T), 0, 6, 8 * sp.cos(T) - 2 * sp.cos(T) ** 2),
        (
            sp.cot(T),
            5 * sp.exp(sp.cos(T)),
            sp.pi / 2,
            -4,
            (1 - 5 * sp.exp(sp.cos(T))) / sp.sin(T),
        ),
    ],
)
def test_q3_la_solution_verifie_l_equation_et_sa_condition(
    p: sp.Expr, q: sp.Expr, depart: sp.Expr, valeur: sp.Expr, attendue: sp.Expr
) -> None:
    """Equations (7), (8) et (9) : la formule du cours, puis la constante."""
    generale = EquationLineaire(p, q).solution_generale()
    constante = sp.solve(sp.Eq(generale.subs(T, depart), valeur), C)
    assert len(constante) == 1
    solution = generale.subs(C, constante[0])
    assert sp.simplify(solution - attendue) == 0
    assert sp.simplify(sp.diff(attendue, T) + p * attendue - q) == 0
    assert sp.simplify(attendue.subs(T, depart) - valeur) == 0


# --- Question 4 : une relation de recurrence --------------------------------


def _i10(n: int) -> sp.Expr:
    return sp.integrate(1 / (X**2 + 1) ** n, (X, 0, 1))


@pytest.mark.parametrize("n", [1, 2, 3])
def test_q4_la_recurrence_de_l_integrale_10(n: int) -> None:
    """`I_(n+1) = (2n - 1) / (2n) I_n + 1 / (n 2^(n+1))`, par parties sur `I_n`."""
    suivante = sp.Rational(2 * n - 1, 2 * n) * _i10(n) + sp.Rational(
        1, n * 2 ** (n + 1)
    )
    assert sp.simplify(_i10(n + 1) - suivante) == 0


def test_q4_i2_de_l_integrale_10() -> None:
    assert sp.simplify(_i10(2) - (sp.pi / 8 + sp.Rational(1, 4))) == 0


@pytest.mark.parametrize("n", [1, 2, 3])
def test_q4_la_recurrence_de_l_integrale_11(n: int) -> None:
    """`I_n = x (ln x)^n - n I_(n-1)` : `u = (ln x)^n`, `dv = dx`."""
    parties = IntegrationParParties(sp.log(POSITIF) ** n, sp.Integer(1))
    precedente = sp.integrate(sp.log(POSITIF) ** (n - 1), POSITIF)
    par_recurrence = POSITIF * sp.log(POSITIF) ** n - n * precedente
    assert _derive_en(par_recurrence, sp.log(POSITIF) ** n, POSITIF)
    assert parties.v() == X


def test_q4_i2_de_l_integrale_11() -> None:
    carre = sp.log(POSITIF) ** 2
    attendue = POSITIF * carre - 2 * POSITIF * sp.log(POSITIF) + 2 * POSITIF
    assert _derive_en(attendue, carre, POSITIF)


def _i12(n: int) -> sp.Expr:
    return sp.integrate(X**n / sp.sqrt(X**2 + 1), (X, 0, 1))


@pytest.mark.parametrize("n", [2, 3, 4])
def test_q4_la_recurrence_de_l_integrale_12_saute_un_rang(n: int) -> None:
    """`n I_n = sqrt(2) - (n - 1) I_(n-2)` : elle relie `I_n` a `I_(n-2)`.

    L'enonce demande une relation « entre I_n et I_(n-1) » ; pour cette
    integrale, l'integration par parties (`u = x^(n-1)`,
    `dv = x dx / sqrt(x^2 + 1)`) en donne une de deux en deux.
    """
    ecart = n * _i12(n) - sp.sqrt(2) + (n - 1) * _i12(n - 2)
    assert abs(sp.N(ecart, 30)) < sp.Float("1e-25")


def test_q4_i2_de_l_integrale_12() -> None:
    """`I_0 = ln(1 + sqrt(2))`, d'ou `I_2 = (sqrt(2) - I_0) / 2`."""
    attendue = (sp.sqrt(2) - sp.log(1 + sp.sqrt(2))) / 2
    assert abs(sp.N(_i12(2) - attendue, 30)) < sp.Float("1e-25")
    assert abs(sp.N(_i12(0) - sp.log(1 + sp.sqrt(2)), 30)) < sp.Float("1e-25")
