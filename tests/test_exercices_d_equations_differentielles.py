"""Mathematique, exercices 2, seance I : les equations differentielles.

Source : `Mathematique/EXS/Exs_02_Equations_differentielles.pdf`, pages 3
a 8. Les reponses officielles sont les oracles ; chacune est reportee dans
son equation -- par derivation implicite quand elle est donnee sous la
forme `g(x, y) = K` -- et la bibliotheque doit reconnaitre le type que la
reponse annonce.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.differentielles.autonomes import (
    EquationAutonome,
    Stabilite,
)
from polytech.maths.domain.differentielles.caracterisation import est_solution
from polytech.maths.domain.differentielles.exactes import FormeDifferentielle
from polytech.maths.domain.differentielles.graphique import isocline
from polytech.maths.domain.differentielles.homogenes import degre_d_homogeneite
from polytech.maths.domain.differentielles.lineaire import EquationLineaire
from polytech.maths.domain.differentielles.premier_ordre import (
    FONCTION,
    INCONNUE,
    VARIABLE,
    EquationDuPremierOrdre,
    Type,
)
from polytech.maths.domain.differentielles.second_ordre import (
    EquationDuSecondOrdre,
)
from polytech.maths.domain.symboles import CONSTANTE as K

X, Y = VARIABLE, INCONNUE
R = sp.Rational
sinh, cosh, sin, cos, exp, sqrt = sp.sinh, sp.cosh, sp.sin, sp.cos, sp.exp, sp.sqrt

SECONDS_MEMBRES = {
    1: X * (Y**2 - 1) / Y,
    2: -X * Y / (X**2 + Y**2),
    3: Y * exp(2 * X) / (1 + exp(2 * X)),
    4: (2 * X * sinh(Y / X) + 3 * Y * cosh(Y / X)) / (3 * X * cosh(Y / X)),
    5: cos(X) * sin(X) - Y * cos(X),
    6: cos(X) - Y * sp.tan(X),
    7: ((2 * X - 1) * Y - X**2) / (X * (X - 1)),
}
"""`y' = f(x, y)` pour chacune des sept equations de l'exercice I.1."""

EXPLICITES = {
    1: sqrt(1 + K * exp(X**2)),
    3: K * sqrt(1 + exp(2 * X)),
    5: sin(X) - 1 + K * exp(-sin(X)),
    6: (K + X) * cos(X),
    7: K * (X**2 - X) + X,
}
IMPLICITES = {
    2: Y * sqrt(Y**2 + 2 * X**2),
    4: sinh(Y / X) / X ** R(2, 3),
}
"""`g(x, y)`, la reponse etant `g(x, y) = K`."""

TYPES = {
    1: Type.VARIABLES_SEPAREES,
    2: Type.HOMOGENE,
    3: Type.VARIABLES_SEPAREES,
    4: Type.HOMOGENE,
    5: Type.LINEAIRE,
    6: Type.LINEAIRE,
    7: Type.LINEAIRE,
}


@pytest.mark.parametrize("numero", list(EXPLICITES))
def test_chaque_reponse_explicite_de_l_exercice_i_1_verifie_son_equation(
    numero: int,
) -> None:
    solution = EXPLICITES[numero]
    reste = sp.diff(solution, X) - SECONDS_MEMBRES[numero].subs(Y, solution)
    assert sp.simplify(reste) == 0


def test_la_premiere_reponse_vaut_avec_les_deux_signes() -> None:
    """`y = +- sqrt(1 + K e^(x^2))` : la racine negative verifie aussi l'equation."""
    solution = -EXPLICITES[1]
    reste = sp.diff(solution, X) - SECONDS_MEMBRES[1].subs(Y, solution)
    assert sp.simplify(reste) == 0


@pytest.mark.parametrize("numero", list(IMPLICITES))
def test_chaque_reponse_implicite_de_l_exercice_i_1_verifie_son_equation(
    numero: int,
) -> None:
    """Le long de `g(x, y) = K`, la pente est `-g_x / g_y`."""
    g = IMPLICITES[numero]
    pente = -sp.diff(g, X) / sp.diff(g, Y)
    assert sp.simplify(pente - SECONDS_MEMBRES[numero]) == 0


@pytest.mark.parametrize("numero", list(TYPES))
def test_la_bibliotheque_reconnait_le_type_que_la_reponse_annonce(numero: int) -> None:
    assert EquationDuPremierOrdre(SECONDS_MEMBRES[numero]).type() is TYPES[numero]


def test_les_deux_equations_homogenes_le_sont_au_degre_zero() -> None:
    assert degre_d_homogeneite(SECONDS_MEMBRES[2]) == 0
    assert degre_d_homogeneite(SECONDS_MEMBRES[4]) == 0


@pytest.mark.parametrize(
    ("numero", "p", "q"),
    [
        (5, cos(X), cos(X) * sin(X)),
        (6, sp.tan(X), cos(X)),
        (7, (1 - 2 * X) / (X * (X - 1)), -X / (X - 1)),
    ],
)
def test_la_variation_des_constantes_retrouve_les_trois_reponses_lineaires(
    numero: int, p: sp.Expr, q: sp.Expr
) -> None:
    """`y' + P y = Q` : la formule (1) du cours doit verifier l'equation."""
    solution = EquationLineaire(p, q).solution_generale()
    reste = sp.diff(solution, X) - SECONDS_MEMBRES[numero].subs(Y, solution)
    assert sp.simplify(reste) == 0


def test_les_isoclines_de_l_exercice_i_2() -> None:
    """`y' = sin(pi y / 4)` : `Gamma_p : y = (4/pi) arcsin p` ; `Gamma_0 : y = 0`."""
    second_membre = sin(sp.pi * Y / 4)
    assert isocline(second_membre, R(1, 2)).subs(Y, 4 / sp.pi * sp.asin(R(1, 2)))
    assert isocline(second_membre, 0).subs(Y, 0)


def test_les_points_critiques_de_l_exercice_i_2() -> None:
    """La figure : pentes nulles en `y = 0` et en `y = +-4` ; les solutions fuient 0."""
    autonome = EquationAutonome(sin(sp.pi * Y / 4))
    assert autonome.nature(0) is Stabilite.INSTABLE
    assert autonome.nature(4) is Stabilite.STABLE
    assert autonome.nature(-4) is Stabilite.STABLE
    assert autonome.sens_de_variation(2) == 1
    assert autonome.sens_de_variation(-2) == -1


def test_la_solution_de_la_logistique_de_l_exercice_i_3() -> None:
    """`y = K c e^(lambda K t) / (1 + c e^(lambda K t))`, `c = y0 / (K - y0)`."""
    lam, capacite, y0 = sp.symbols("lambda kappa y_0", positive=True)
    c = y0 / (capacite - y0)
    solution = (
        capacite * c * exp(lam * capacite * X) / (1 + c * exp(lam * capacite * X))
    )
    reste = sp.diff(solution, X) - lam * solution * (capacite - solution)
    assert sp.simplify(reste) == 0
    assert sp.simplify(solution.subs(X, 0) - y0) == 0


def test_les_points_critiques_de_la_logistique() -> None:
    """Page 6 : `y = 0` est un point critique instable, `y = 1` un point stable."""
    autonome = EquationAutonome(Y * (1 - Y))
    assert autonome.points_critiques() == (0, 1)
    assert autonome.nature(0) is Stabilite.INSTABLE
    assert autonome.nature(1) is Stabilite.STABLE


def test_le_parametre_qui_rend_l_equation_exacte() -> None:
    """Exercice I.4.1 : `2x + alpha y + 2x e^(x^2) + (2y + x) y' = 0`, `alpha = 1`."""
    alpha = sp.Symbol("alpha", real=True)
    forme = FormeDifferentielle(2 * X + alpha * Y + 2 * X * exp(X**2), 2 * Y + X)
    assert sp.solve(forme.ecart_de_schwarz(), alpha) == [1]
    exacte = FormeDifferentielle(2 * X + Y + 2 * X * exp(X**2), 2 * Y + X)
    assert exacte.est_exacte()


def test_la_solution_implicite_de_l_exercice_i_4() -> None:
    """`g(x, y) = sin(x + y) + 2 y^2 + x^2 = c`."""
    forme = FormeDifferentielle(cos(X + Y) + 2 * X, cos(X + Y) + 4 * Y)
    assert forme.est_exacte()
    assert sp.simplify(forme.potentiel() - (sin(X + Y) + 2 * Y**2 + X**2)) == 0


def test_le_systeme_masse_ressort_hors_resonance() -> None:
    """Exercice I.5 : `y'' + y = sin(w t)`, particuliere `sin(w t)/(1 - w^2)`."""
    omega = sp.Rational(3)
    c1, c2 = sp.symbols("c_1 c_2")
    y = FONCTION(X)
    equation = sp.Eq(y.diff(X, 2) + y, sin(omega * X))
    officielle = c1 * cos(X) + c2 * sin(X) + sin(omega * X) / (1 - omega**2)
    assert est_solution(equation, officielle)
    oscillateur = EquationDuSecondOrdre(1, 0, 1)
    assert not oscillateur.est_en_resonance(omega)
    particuliere = oscillateur.solution_particuliere(sin(omega * X))
    assert sp.simplify(particuliere - sin(omega * X) / (1 - omega**2)) == 0


def test_le_systeme_masse_ressort_en_resonance() -> None:
    """`w = 1` : `-(t/2) cos t`, dont l'amplitude croit avec le temps."""
    y = FONCTION(X)
    equation = sp.Eq(y.diff(X, 2) + y, sin(X))
    assert est_solution(equation, -X / 2 * cos(X))
    oscillateur = EquationDuSecondOrdre(1, 0, 1)
    assert oscillateur.est_en_resonance(1)
    particuliere = oscillateur.solution_particuliere(sin(X))
    assert sp.simplify(particuliere + X / 2 * cos(X)) == 0


def test_la_solution_homogene_du_systeme_masse_ressort() -> None:
    """`y'' + y = 0` : racines `+-j`, donc `c1 cos t + c2 sin t`."""
    oscillateur = EquationDuSecondOrdre(1, 0, 1)
    assert set(oscillateur.racines_caracteristiques()) == {sp.I, -sp.I}
    assert oscillateur.pulsation_propre() == 1
    y = FONCTION(X)
    assert est_solution(sp.Eq(y.diff(X, 2) + y, 0), oscillateur.solution_homogene())


def test_des_racines_reelles_ne_donnent_pas_de_pulsation_propre() -> None:
    """`y'' + 3 y' + 2 y = 0` : racines -1 et -2, aucun regime oscillant."""
    amorti = EquationDuSecondOrdre(1, 3, 2)
    assert set(amorti.racines_caracteristiques()) == {-1, -2}
    assert amorti.pulsation_propre() is None
    assert not amorti.est_en_resonance(1)
    y = FONCTION(X)
    equation = sp.Eq(y.diff(X, 2) + 3 * y.diff(X) + 2 * y, 0)
    assert est_solution(equation, amorti.solution_homogene())


def test_une_equation_sans_terme_du_second_ordre_est_refusee() -> None:
    with pytest.raises(ValueError, match="coefficient de y'' nul"):
        EquationDuSecondOrdre(0, 1, 1)


def test_une_equation_separable_qui_s_annule_en_y_egal_un_est_reconnue() -> None:
    """`y' = x (y^2 - 1) / y` : `g(1) = 0`, `f(x)` se lit ailleurs qu'en `y = 1`."""
    equation = EquationDuPremierOrdre(X * (Y**2 - 1) / Y)
    assert equation.type() is Type.VARIABLES_SEPAREES
    inverse_de_g, en_x = equation.membres_separes()
    assert sp.simplify(en_x * (1 / inverse_de_g) - X * (Y**2 - 1) / Y) == 0


def test_un_second_membre_nul_n_est_d_aucun_type() -> None:
    """`y' = 0` : aucune sonde ne rend un `f(x)` non nul."""
    with pytest.raises(ValueError, match="equation non separable"):
        EquationDuPremierOrdre(sp.Integer(0) * Y).membres_separes()


def test_une_racine_double_donne_un_facteur_affine() -> None:
    """`y'' + 2 y' + y = 0` (exercice II.5.2) : `(c1 + c2 t) e^(-t)`."""
    critique = EquationDuSecondOrdre(1, 2, 1)
    assert critique.racines_caracteristiques() == (-1, -1)
    c1, c2 = sp.symbols("c_1 c_2")
    assert sp.simplify(critique.solution_homogene() - (c1 + c2 * X) * exp(-X)) == 0
    y = FONCTION(X)
    equation = sp.Eq(y.diff(X, 2) + 2 * y.diff(X) + y, 0)
    assert est_solution(equation, critique.solution_homogene())
