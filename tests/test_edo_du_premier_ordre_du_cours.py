"""Mathematique, second jeu, planches 20 a 40 : les EDO du premier ordre.

Oracles (`Mathematique/SLIDES/Slides_02_Equations_differentielles.pdf`) :
l'equation exacte de la planche 21, `g = x^3/3 + x y - y^2` ; `y' = -2xy`
et ses constantes (planche 23) ; le champ des tangentes de `y' = x - y^2`
(planches 26-27) ; `y' = y^2` et ses solutions maximales (planche 31) ;
`y' = y^2 - y - 6` et ses equilibres (planche 35) ; `y' = 2xy/(x^2 - y^2)`
(planche 37) ; la formule (1) des equations lineaires (planche 39).
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.differentielles.autonomes import (
    EquationAutonome,
    Stabilite,
)
from polytech.maths.domain.differentielles.exactes import FormeDifferentielle
from polytech.maths.domain.differentielles.graphique import (
    isocline,
    regions_de_variation,
    tangente,
)
from polytech.maths.domain.differentielles.homogenes import (
    Z,
    degre_d_homogeneite,
    equation_en_z,
)
from polytech.maths.domain.differentielles.lineaire import EquationLineaire
from polytech.maths.domain.differentielles.premier_ordre import INCONNUE, VARIABLE
from polytech.maths.domain.differentielles.prolongement import (
    est_globale,
    intervalles_de_definition,
)
from polytech.maths.domain.differentielles.separables import (
    resoudre_par_separation,
    solutions_constantes,
)
from polytech.maths.domain.symboles import CONSTANTE as C

X, Y = VARIABLE, INCONNUE
R = sp.Rational
SOURCE = "Mathematique/SLIDES/Slides_02_Equations_differentielles.pdf"


def _verifie(solution: sp.Expr, second_membre: sp.Expr) -> bool:
    """`y' = f(x, y)` le long de la solution proposee."""
    reste = sp.diff(solution, X) - second_membre.subs(Y, solution)
    return bool(sp.simplify(reste) == 0)


def test_l_equation_exacte_de_la_planche_21() -> None:
    """`(x^2 + y) dx + (x - 2y) dy = 0`, potentiel `x^3/3 + x y - y^2`."""
    forme = FormeDifferentielle(X**2 + Y, X - 2 * Y)
    assert forme.est_exacte()
    assert sp.simplify(forme.potentiel() - (X**3 / 3 + X * Y - Y**2)) == 0


def test_le_potentiel_a_bien_les_deux_derivees_partielles_demandees() -> None:
    """Planche 21 : `dg/dx = x^2 + y`, `dg/dy = x - 2y`."""
    potentiel = FormeDifferentielle(X**2 + Y, X - 2 * Y).potentiel()
    assert sp.diff(potentiel, X) == X**2 + Y
    assert sp.diff(potentiel, Y) == X - 2 * Y


def test_la_separation_des_variables_de_la_planche_23() -> None:
    """`y' = -2xy` : `ln|y| = -x^2 + K`, d'ou `y = K e^(-x^2)`."""
    raisonnement = resoudre_par_separation(-2 * X, Y)
    assert [e.affirmation for e in raisonnement.etapes] == [
        "dy/y = -2*x dx",
        "log(Abs(y)) = -x**2 + K_1",
        "y = C*exp(-x**2)",
        "y = 0 est aussi solution",
    ]
    assert raisonnement.sources == (f"{SOURCE}, planches 22-23",)
    assert raisonnement.conclusion == "y = C*exp(-x**2), C reel quelconque"


def test_la_solution_obtenue_par_separation_verifie_l_equation() -> None:
    assert _verifie(C * sp.exp(-(X**2)), -2 * X * Y)


def test_les_solutions_constantes_sont_les_zeros_de_g() -> None:
    """Planche 23, « attention aux constantes » : diviser par `g(y)` les perd."""
    assert solutions_constantes(Y) == (0,)
    assert solutions_constantes(Y * (Y - 3)) == (0, 3)
    assert solutions_constantes(Y**2 + 1) == ()


def test_sans_solution_constante_la_separation_s_arrete_a_la_forme_implicite() -> None:
    """`y' = x (1 + y^2)` : `arctan y = x^2/2 + K`, sans solution constante."""
    raisonnement = resoudre_par_separation(X, 1 + Y**2)
    assert raisonnement.etapes[1].affirmation == "atan(y) = x**2/2 + K_1"
    assert len(raisonnement.etapes) == 3
    assert _verifie(sp.tan(X**2 / 2 + C), X * (1 + Y**2))


def test_la_tangente_au_champ_en_un_point() -> None:
    """Planche 25 : `D_M : y - y0 = f(x0, y0) (x - x0)` ; `f = x - y^2` en `(2, 1)`."""
    assert tangente(X - Y**2, (2, 1)) == sp.Eq(Y - 1, 1 * (X - 2))
    assert tangente(X - Y**2, (0, 0)) == sp.Eq(Y, 0)


def test_les_isoclines_de_la_planche_27_sont_des_paraboles() -> None:
    """`Gamma_p : x - y^2 = p` ; `Gamma_{3/2}` passe par `(3/2, 0)` et `(5/2, 1)`."""
    courbe = isocline(X - Y**2, R(3, 2))
    assert courbe == sp.Eq(X - Y**2, R(3, 2))
    assert courbe.subs({X: R(3, 2), Y: 0})
    assert courbe.subs({X: R(5, 2), Y: 1})


def test_l_isocline_zero_separe_les_solutions_croissantes_des_decroissantes() -> None:
    """Planche 27 : `y' > 0` a droite de la parabole `x = y^2`, `y' < 0` a gauche."""
    croissante, decroissante = regions_de_variation(X - Y**2)
    assert croissante.subs({X: 2, Y: 1})
    assert not croissante.subs({X: 0, Y: 1})
    assert decroissante.subs({X: 0, Y: 1})
    assert not decroissante.subs({X: 1, Y: 1})


def test_la_courbe_integrale_de_la_planche_28() -> None:
    """`y' = -2xy` : `y = K e^(-x^2)` a en chaque point la pente du champ."""
    courbe = C * sp.exp(-(X**2))
    pente_du_champ = (-2 * X * Y).subs(Y, courbe)
    assert sp.simplify(sp.diff(courbe, X) - pente_du_champ) == 0


def test_les_solutions_maximales_de_la_planche_31() -> None:
    """`y' = y^2` : `y = -1/(t + C)` vit sur `]-oo, -C[` ou sur `]-C, +oo[`."""
    solution = -1 / (X + 2)
    assert _verifie(solution, Y**2)
    assert intervalles_de_definition(solution) == (
        sp.Interval.open(-sp.oo, -2),
        sp.Interval.open(-2, sp.oo),
    )
    assert not est_globale(solution)


def test_seule_la_solution_nulle_est_globale() -> None:
    """Planche 31 : « seule `y(t) = 0` est globale dans ce cas »."""
    assert _verifie(sp.Integer(0), Y**2)
    assert est_globale(sp.Integer(0))
    assert intervalles_de_definition(sp.Integer(0)) == (sp.S.Reals,)


def test_les_points_critiques_de_la_planche_32() -> None:
    """`y' = y (a y - b)` : `c = 0` et `c = b/a`."""
    a, b = sp.symbols("a b", positive=True)
    assert set(EquationAutonome(Y * (a * Y - b)).points_critiques()) == {0, b / a}


def test_les_equilibres_de_la_planche_35() -> None:
    """`y' = (y - 3)(y + 2)` : `-2` stable, `3` instable."""
    equation = EquationAutonome(Y**2 - Y - 6)
    assert equation.points_critiques() == (-2, 3)
    assert equation.nature(-2) is Stabilite.STABLE
    assert equation.nature(3) is Stabilite.INSTABLE


def test_un_point_critique_peut_etre_semi_stable() -> None:
    """Planche 34 : croissantes (ou decroissantes) des deux cotes ; `y' = y^2`."""
    assert EquationAutonome(Y**2).nature(0) is Stabilite.SEMI_STABLE
    assert EquationAutonome(-((Y - 1) ** 2)).nature(1) is Stabilite.SEMI_STABLE


def test_un_point_qui_n_est_pas_critique_n_a_pas_de_nature() -> None:
    with pytest.raises(ValueError, match="n'est pas un point critique"):
        EquationAutonome(Y**2 - Y - 6).nature(0)


def test_entre_deux_equilibres_les_solutions_sont_monotones() -> None:
    """Planche 33 : croissante, decroissante ou constante ; ici entre -2 et 3."""
    equation = EquationAutonome(Y**2 - Y - 6)
    assert equation.sens_de_variation(0) == -1
    assert equation.sens_de_variation(4) == 1
    assert equation.sens_de_variation(-3) == 1
    assert equation.sens_de_variation(3) == 0


def test_le_degre_d_homogeneite_d_une_fonction() -> None:
    """Planche 36 : `f(lambda x, lambda y) = lambda^n f(x, y)`."""
    assert degre_d_homogeneite(X**2 + Y**2) == 2
    assert degre_d_homogeneite(2 * X * Y / (X**2 - Y**2)) == 0
    assert degre_d_homogeneite(X + Y**2) is None


def test_l_equation_homogene_de_la_planche_37() -> None:
    """`y' = 2xy/(x^2 - y^2)` : avec `z = y/x`, `f = 2z/(1 - z^2)` et `x z' = f - z`."""
    fonction_de_z, pente_de_z = equation_en_z(2 * X * Y / (X**2 - Y**2))
    assert sp.simplify(fonction_de_z - 2 * Z / (1 - Z**2)) == 0
    assert sp.simplify(pente_de_z - (Z + Z**3) / (1 - Z**2)) == 0


def test_une_equation_qui_n_est_pas_homogene_de_degre_zero_est_refusee() -> None:
    with pytest.raises(ValueError, match="pas homogene de degre 0"):
        equation_en_z(X + Y)


def test_la_solution_de_l_equation_homogene_de_la_planche_37() -> None:
    """Apres separation en `z` : `y = C (x^2 + y^2)`, des cercles tangents a l'axe."""
    second_membre = 2 * X * Y / (X**2 - Y**2)
    implicite = Y - C * (X**2 + Y**2)
    pente = -sp.diff(implicite, X) / sp.diff(implicite, Y)
    sur_la_courbe = sp.solve(implicite, C)[0]
    assert sp.simplify((pente - second_membre).subs(C, sur_la_courbe)) == 0


def test_la_formule_generale_de_l_equation_lineaire() -> None:
    """Planche 39, formule (1), sur l'exercice de la planche 24 : `y' + 2y = 3`."""
    equation = EquationLineaire(sp.Integer(2), sp.Integer(3))
    assert sp.simplify(equation.solution_homogene() - C * sp.exp(-2 * X)) == 0
    assert sp.simplify(equation.solution_particuliere() - R(3, 2)) == 0
    attendu = R(3, 2) + C * sp.exp(-2 * X)
    assert sp.simplify(equation.solution_generale() - attendu) == 0
    assert _verifie(equation.solution_generale(), 3 - 2 * Y)


def test_la_variation_des_constantes_avec_un_coefficient_variable() -> None:
    """Planche 40 : `y' + y/x = x^2`, `K(x) = int x^2 . x dx = x^4/4`."""
    equation = EquationLineaire(1 / X, X**2)
    assert sp.simplify(equation.constante_variable() - X**4 / 4) == 0
    assert sp.simplify(equation.solution_particuliere() - X**3 / 4) == 0
    assert _verifie(equation.solution_generale(), X**2 - Y / X)


def test_la_solution_generale_est_la_somme_des_deux() -> None:
    """Planche 40 : `y_SGENH = y_SGEH + y_SPENH`."""
    equation = EquationLineaire(1 / X, X**2)
    somme = equation.solution_homogene() + equation.solution_particuliere()
    assert sp.simplify(equation.solution_generale() - somme) == 0


def test_l_explication_de_la_variation_des_constantes() -> None:
    raisonnement = EquationLineaire(sp.Integer(2), sp.Integer(3)).expliquer()
    assert [e.affirmation for e in raisonnement.etapes] == [
        "y_SGEH = C*exp(-2*x)",
        "K(x) = 3*exp(2*x)/2",
        "y_SPENH = 3/2",
    ]
    assert raisonnement.conclusion == "y = C*exp(-2*x) + 3/2"
    assert raisonnement.sources == (f"{SOURCE}, planches 38-40",)


def test_une_fonction_plate_en_son_point_critique_n_est_pas_classee() -> None:
    """`f = 0` partout : toutes les derivees sont nulles, rien ne decide."""
    with pytest.raises(ValueError, match="f est plate"):
        EquationAutonome(sp.Integer(0) * Y).nature(1)


def test_une_solution_constante_hors_de_la_famille_reste_a_part() -> None:
    """`y' = y^2` : `y = -1/(C + x)` ne rend `y = 0` pour aucun `C` fini."""
    raisonnement = resoudre_par_separation(sp.Integer(1), Y**2)
    assert raisonnement.etapes[-1].affirmation == "y = 0 est aussi solution"
    assert raisonnement.conclusion == "y = -1/(C + x)"
    sans_constante = resoudre_par_separation(X, 1 + Y**2)
    assert sans_constante.conclusion == "y = tan(C + x**2/2)"
