"""Syllabus de mathematiques appliquees, chapitre 1, sections 1.1 a 1.3.

Source : `Mathematique/SYLLABUS/Syllabus_Mathematiques_appliquees_2016_2017.pdf`,
pages 4 a 17. Ce que le syllabus ajoute aux planches : les singularites
des equations (1.4) et (1.5), le pendule linearise (1.6)-(1.7), le facteur
integrant (exemple 1.4), la cuve d'alcool (exemple 1.5), l'equation de
Bernoulli (exemple 1.6). Les exemples resolus sont les oracles.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.chemistry.domain.melanges import (
    concentration_dans_la_cuve,
    equation_de_la_cuve,
)
from polytech.maths.domain.differentielles.bernoulli import EquationDeBernoulli
from polytech.maths.domain.differentielles.caracterisation import caracteriser
from polytech.maths.domain.differentielles.exactes import FormeDifferentielle
from polytech.maths.domain.differentielles.facteur_integrant import (
    facteur_integrant,
    rendre_exacte,
)
from polytech.maths.domain.differentielles.modeles import pendule
from polytech.maths.domain.differentielles.premier_ordre import (
    FONCTION,
    INCONNUE,
    VARIABLE,
)
from polytech.maths.domain.differentielles.prolongement import (
    intervalles_de_definition,
)
from polytech.maths.domain.differentielles.second_ordre import (
    EquationDuSecondOrdre,
)
from polytech.maths.domain.symboles import CONSTANTE as K

X, Y = VARIABLE, INCONNUE
R = sp.Rational


def _verifie(solution: sp.Expr, second_membre: sp.Expr) -> bool:
    reste = sp.diff(solution, X) - second_membre.subs(Y, solution)
    return bool(sp.simplify(reste) == 0)


def test_la_singularite_d_une_equation_lineaire_se_lit_sur_l_equation() -> None:
    """(1.4) : `y' + y/(x - 1) = 0`, `y = y(0)/(1 - x)` -- singuliere en `x = 1`."""
    solution = 5 / (1 - X)
    assert _verifie(solution, -Y / (X - 1))
    assert solution.subs(X, 0) == 5
    bornes = {i.sup for i in intervalles_de_definition(solution)} - {sp.oo}
    assert bornes == {1}


def test_la_singularite_non_lineaire_depend_de_la_condition_initiale() -> None:
    """(1.5) : `y' = y^2`, `y = y(0)/(1 - y(0) x)` -- singuliere en `x = 1/y(0)`."""
    for initiale in (2, 5):
        solution = initiale / (1 - initiale * X)
        assert _verifie(solution, Y**2)
        assert solution.subs(X, 0) == initiale
        bornes = {i.sup for i in intervalles_de_definition(solution)} - {sp.oo}
        assert bornes == {R(1, initiale)}


def test_le_pendule_n_est_lineaire_que_pour_les_petits_mouvements() -> None:
    """(1.6) et (1.7) : `sin theta` y est remplace par `theta`."""
    g, longueur = sp.symbols("g L", positive=True)
    exacte, linearisee = pendule(g, longueur)
    theta = FONCTION(X)
    assert exacte == sp.Eq(theta.diff(X, 2) + g / longueur * sp.sin(theta), 0)
    assert linearisee == sp.Eq(theta.diff(X, 2) + g / longueur * theta, 0)
    assert not caracteriser(exacte).lineaire
    assert caracteriser(linearisee).lineaire


def test_le_pendule_linearise_oscille_a_la_pulsation_racine_de_g_sur_l() -> None:
    g, longueur = sp.symbols("g L", positive=True)
    oscillateur = EquationDuSecondOrdre(1, 0, g / longueur)
    assert oscillateur.pulsation_propre() == sp.sqrt(g / longueur)


def test_l_equation_de_l_exemple_1_4_n_est_pas_exacte() -> None:
    """`P = 3xy^2 + 2y`, `Q = 2x^2 y + x` : `dQ/dx = 4xy + 1`, `dP/dy = 6xy + 2`."""
    forme = FormeDifferentielle(3 * X * Y**2 + 2 * Y, 2 * X**2 * Y + X)
    assert not forme.est_exacte()
    assert sp.diff(forme.facteur_en_y, X) == 4 * X * Y + 1
    assert sp.diff(forme.facteur_en_x, Y) == 6 * X * Y + 2


def test_le_facteur_integrant_de_l_exemple_1_4_est_x() -> None:
    """`dh/h = dx/x`, donc `h(x) = x`."""
    forme = FormeDifferentielle(3 * X * Y**2 + 2 * Y, 2 * X**2 * Y + X)
    assert facteur_integrant(forme) == X


def test_multipliee_par_son_facteur_integrant_l_equation_devient_exacte() -> None:
    """Exemple 1.4 : apres multiplication par `x`, `x^3 y^2 + x^2 y = C`."""
    forme = FormeDifferentielle(3 * X * Y**2 + 2 * Y, 2 * X**2 * Y + X)
    exacte = rendre_exacte(forme)
    assert sp.expand(exacte.facteur_en_x) == 3 * X**2 * Y**2 + 2 * X * Y
    assert sp.expand(exacte.facteur_en_y) == 2 * X**3 * Y + X**2
    assert exacte.est_exacte()
    assert sp.simplify(exacte.potentiel() - (X**3 * Y**2 + X**2 * Y)) == 0


def test_un_facteur_integrant_peut_ne_dependre_que_de_y() -> None:
    """Le second cas de la page 12 : `y dx + (2x - y e^y) dy = 0`, `h(y) = y`."""
    forme = FormeDifferentielle(Y, 2 * X - Y * sp.exp(Y))
    assert facteur_integrant(forme) == Y
    assert rendre_exacte(forme).est_exacte()


def test_une_forme_deja_exacte_a_pour_facteur_integrant_un() -> None:
    assert facteur_integrant(FormeDifferentielle(X**2 + Y, X - 2 * Y)) == 1


def test_sans_facteur_en_x_seul_ni_en_y_seul_la_recherche_le_dit() -> None:
    """`(xy + y^3) dx + (x^3 + xy) dy = 0` : aucun des deux quotients ne convient."""
    forme = FormeDifferentielle(X * Y + Y**3, X**3 + X * Y)
    with pytest.raises(ValueError, match="ni de x seul ni de y seul"):
        facteur_integrant(forme)


def test_la_cuve_d_alcool_de_l_exemple_1_5() -> None:
    """`V C' = q - q C`, `C(0) = 0` : `C(t) = 1 - e^(-q t / V)`, qui tend vers 1."""
    q, volume = sp.symbols("q V", positive=True)
    concentration = concentration_dans_la_cuve(
        q, volume, entree=1, initiale=0, instant=X
    )
    assert sp.simplify(concentration - (1 - sp.exp(-q * X / volume))) == 0
    assert sp.limit(concentration, X, sp.oo) == 1


def test_la_cuve_est_une_equation_lineaire_du_premier_ordre() -> None:
    """Exemple 1.5 : `C' + (q/V) C = q/V`, de solution generale `1 + C e^(-q t/V)`."""
    q, volume = sp.symbols("q V", positive=True)
    equation = equation_de_la_cuve(q, volume, entree=1)
    attendue = 1 + K * sp.exp(-q * X / volume)
    assert sp.simplify(equation.solution_generale() - attendue) == 0


def test_la_cuve_tend_vers_la_concentration_d_entree() -> None:
    """Une entree a 30 %, une cuve deja a 80 % : la concentration descend vers 0,3."""
    concentration = concentration_dans_la_cuve(
        2, 10, entree=R(3, 10), initiale=R(4, 5), instant=X
    )
    assert concentration.subs(X, 0) == R(4, 5)
    assert sp.limit(concentration, X, sp.oo) == R(3, 10)
    assert sp.diff(concentration, X).subs(X, 1) < 0


def test_l_equation_de_bernoulli_de_l_exemple_1_6() -> None:
    """`y' + y/(x + 1) = -y^2` : `z = 1/y`, `z' - z/(x + 1) = 1`."""
    equation = EquationDeBernoulli(1 / (X + 1), sp.Integer(-1), 2)
    lineaire = equation.equation_en_z()
    assert sp.simplify(lineaire.p + 1 / (X + 1)) == 0
    assert lineaire.q == 1
    assert sp.simplify(lineaire.solution_homogene() - K * (X + 1)) == 0


def test_la_solution_generale_de_l_exemple_1_6() -> None:
    """`y = 1 / ((x + 1)(K + ln|x + 1|))`."""
    equation = EquationDeBernoulli(1 / (X + 1), sp.Integer(-1), 2)
    attendue = 1 / ((X + 1) * (K + sp.log(X + 1)))
    assert sp.simplify(equation.solution_generale() - attendue) == 0
    assert _verifie(equation.solution_generale(), -(Y**2) - Y / (X + 1))


def test_la_solution_singuliere_d_une_equation_de_bernoulli() -> None:
    """Remarque 1.1 : diviser par `y^a` elimine `y = 0`, a ne pas oublier."""
    equation = EquationDeBernoulli(1 / (X + 1), sp.Integer(-1), 2)
    assert equation.solution_singuliere() == 0
    assert _verifie(sp.Integer(0), -(Y**2) - Y / (X + 1))


@pytest.mark.parametrize("exposant", [0, 1])
def test_un_exposant_zero_ou_un_n_est_pas_une_equation_de_bernoulli(
    exposant: int,
) -> None:
    """(1.24) : `a` different de 0 et de 1 -- sinon l'equation est deja lineaire."""
    with pytest.raises(ValueError, match="equation deja lineaire"):
        EquationDeBernoulli(X, X, exposant)


def test_le_retour_a_y_eleve_z_a_la_puissance_un_sur_un_moins_a() -> None:
    """`y' + y = x sqrt(y)`, `a = 1/2` : `z = sqrt(y)`, `z' + z/2 = x/2`, `y = z^2`.

    `z = K e^(-x/2) + x - 2`, et `y' + y = z (2 z' + z) = x z`.
    """
    equation = EquationDeBernoulli(sp.Integer(1), X, R(1, 2))
    z = K * sp.exp(-X / 2) + X - 2
    assert sp.simplify(equation.equation_en_z().solution_generale() - z) == 0
    assert sp.simplify(equation.solution_generale() - z**2) == 0
    assert sp.simplify(sp.diff(z**2, X) + z**2 - X * z) == 0
