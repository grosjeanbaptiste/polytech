"""Mathematique, second jeu, planches 2 a 19 : les EDO et le probleme de Cauchy.

Oracles (`Mathematique/SLIDES/Slides_02_Equations_differentielles.pdf`) :
les trois equations de la planche 9 ; `y'' + y = 0` et ses solutions
`cos x`, `2 sin x`, `a cos x + b sin x` (planche 10) ; l'equation
integrale de la planche 14 ; le seau troue et la datation au carbone 14
(planches 16 a 19), l'un a une infinite de solutions, l'autre a une seule.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.differentielles.caracterisation import (
    caracteriser,
    est_solution,
    forme_normale,
)
from polytech.maths.domain.differentielles.cauchy import ProblemeDeCauchy
from polytech.maths.domain.differentielles.premier_ordre import (
    FONCTION,
    INCONNUE,
    VARIABLE,
)

X, Y = VARIABLE, INCONNUE
y = FONCTION(X)
A, B = sp.symbols("a b", real=True)
GAMMA = sp.Symbol("gamma", negative=True)
BETA = sp.Symbol("beta", positive=True)


def test_la_premiere_equation_de_la_planche_9() -> None:
    """`y'' + 5 y' + 4 y = 3` : ordre 2, lineaire, a coefficients constants."""
    equation = sp.Eq(y.diff(X, 2) + 5 * y.diff(X) + 4 * y, 3)
    caracterisation = caracteriser(equation)
    assert caracterisation.ordre == 2
    assert caracterisation.lineaire
    assert caracterisation.coefficients_constants


def test_la_deuxieme_equation_de_la_planche_9_n_est_pas_lineaire() -> None:
    """`y'' + 5 y' + 4 y^2 = 0` : le carre de `y` rompt la linearite."""
    caracterisation = caracteriser(sp.Eq(y.diff(X, 2) + 5 * y.diff(X) + 4 * y**2, 0))
    assert caracterisation.ordre == 2
    assert not caracterisation.lineaire
    assert not caracterisation.coefficients_constants


def test_la_troisieme_equation_de_la_planche_9_a_des_coefficients_variables() -> None:
    """`x^2 y''' + 5 cos(x) y' + 4 y = e^x` : ordre 3, lineaire, coefficients en `x`."""
    equation = sp.Eq(X**2 * y.diff(X, 3) + 5 * sp.cos(X) * y.diff(X) + 4 * y, sp.exp(X))
    caracterisation = caracteriser(equation)
    assert caracterisation.ordre == 3
    assert caracterisation.lineaire
    assert not caracterisation.coefficients_constants


def test_un_produit_de_y_par_sa_derivee_n_est_pas_lineaire() -> None:
    assert not caracteriser(sp.Eq(y * y.diff(X), 1)).lineaire
    assert not caracteriser(sp.Eq(sp.sin(y) + y.diff(X), 0)).lineaire


def test_une_equation_sans_derivee_n_est_pas_differentielle() -> None:
    with pytest.raises(ValueError, match="aucune derivee"):
        caracteriser(sp.Eq(y, X))


def test_les_solutions_de_la_planche_10() -> None:
    """`y'' + y = 0` : `cos x`, `2 sin x`, et la solution generale."""
    equation = sp.Eq(y.diff(X, 2) + y, 0)
    assert est_solution(equation, sp.cos(X))
    assert est_solution(equation, 2 * sp.sin(X))
    assert est_solution(equation, A * sp.cos(X) + B * sp.sin(X))
    assert not est_solution(equation, sp.exp(X))


def test_la_forme_normale_isole_la_derivee_d_ordre_le_plus_eleve() -> None:
    """Planche 8 : `y^(n) = f[x, y, ..., y^(n-1)]`."""
    equation = sp.Eq(X**2 * y.diff(X, 3) + 5 * sp.cos(X) * y.diff(X) + 4 * y, sp.exp(X))
    attendu = (sp.exp(X) - 5 * sp.cos(X) * y.diff(X) - 4 * y) / X**2
    assert sp.simplify(forme_normale(equation) - attendu) == 0


def test_un_probleme_de_cauchy_joint_une_condition_a_l_equation() -> None:
    """Planche 13 : `y' = f(x, y)`, `y(x0) = y0`. Ici `y' = y`, `y(0) = 1`."""
    probleme = ProblemeDeCauchy(Y, x0=0, y0=1)
    assert probleme.est_solution(sp.exp(X))
    assert not probleme.est_solution(2 * sp.exp(X))
    assert not probleme.est_solution(1 + X)


def test_l_equation_integrale_se_resout_par_approximations_successives() -> None:
    """Planche 14 : `y(x) = y0 + int f(u, y(u)) du`, point fixe d'un operateur.

    Pour `y' = y`, `y(0) = 1`, les iterees sont les sommes partielles de
    la serie de l'exponentielle.
    """
    iterees = ProblemeDeCauchy(Y, x0=0, y0=1).iterees_de_picard(3)
    assert iterees == (1, 1 + X, 1 + X + X**2 / 2, 1 + X + X**2 / 2 + X**3 / 6)


def test_les_iterees_partent_de_la_condition_initiale() -> None:
    """`y' = 2x`, `y(1) = 3` : `3 + (x^2 - 1)` des la premiere iteree."""
    iterees = ProblemeDeCauchy(2 * X, x0=1, y0=3).iterees_de_picard(2)
    assert iterees[0] == 3
    assert sp.expand(iterees[1]) == X**2 + 2
    assert iterees[2] == iterees[1]


def test_la_solution_exacte_verifie_l_equation_integrale() -> None:
    probleme = ProblemeDeCauchy(Y, x0=0, y0=1)
    assert probleme.verifie_l_equation_integrale(sp.exp(X))
    assert not probleme.verifie_l_equation_integrale(1 + X)


def test_la_datation_a_une_solution_unique() -> None:
    """Planche 17 : `x' = -beta x`, `x(t) = x(t0) exp(-beta (t - t0))`."""
    t0, x0 = sp.symbols("t_0 x_0", positive=True)
    probleme = ProblemeDeCauchy(-BETA * Y, x0=t0, y0=x0)
    assert probleme.est_solution(x0 * sp.exp(-BETA * (X - t0)))
    assert sp.simplify(probleme.solution() - x0 * sp.exp(-BETA * (X - t0))) == 0


@pytest.mark.parametrize("alpha", [0, 1, sp.Rational(5, 2), 3])
def test_le_seau_troue_a_une_infinite_de_solutions(alpha: sp.Expr) -> None:
    """Planche 17 : `h' = gamma sqrt(h)`, `h(t1) = 0` -- une solution par `alpha <= t1`.

    La planche n'ecrit pas que `gamma < 0`. C'est pourtant ce que demande
    sa solution : la hauteur decroit jusqu'a `alpha`, puis reste nulle.
    """
    t1 = 3
    avant = GAMMA**2 / 4 * (X - alpha) ** 2
    hauteur = sp.Piecewise((avant, alpha >= X), (0, True))
    probleme = ProblemeDeCauchy(GAMMA * sp.sqrt(Y), x0=t1, y0=0)
    assert probleme.est_solution(hauteur)


def test_avec_gamma_positif_la_solution_imprimee_ne_convient_pas() -> None:
    """Le signe que la planche tait : avec `gamma > 0`, l'equation est fausse."""
    gamma = sp.Symbol("g", positive=True)
    avant_alpha = gamma**2 / 4 * (X - 1) ** 2
    reste = sp.diff(avant_alpha, X) - gamma * sp.sqrt(avant_alpha)
    assert sp.simplify(reste.subs(X, 0)) == -(gamma**2)
