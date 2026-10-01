"""Integrales definies, integrales impropres, et transformee de Laplace.

Deux ancrages viennent du cours lui-meme.

Le fascicule `Exs_01` publie une integrale definie en clair :
`int_0^{pi/2} cos^4 t dt = 3 pi / 16`. C'est la seule de la section 1.2
que l'OCR restitue, et elle sert ici de point fixe.

Les planches `Slides_03_Transformee_de_Laplace` notent la variable image
`p`, et non `s`. Le detail n'en est pas un : un corrige qui ecrirait `s`
obligerait le lecteur a traduire a chaque ligne.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.integration.definie import IntegraleDefinie
from polytech.maths.domain.integration.impropres import IntegraleImpropre
from polytech.maths.domain.laplace.resolution import ProblemeDeCauchy
from polytech.maths.domain.laplace.transformee import Transformee
from polytech.maths.domain.symboles import IMAGE, TEMPS
from polytech.maths.domain.symboles import VARIABLE as X


def test_l_integrale_definie_du_fascicule_vaut_trois_pi_sur_seize() -> None:
    """Section 1.2 du fascicule : la seule valeur que l'OCR en restitue."""
    integrale = IntegraleDefinie(
        integrande=sp.cos(X) ** 4,
        borne_basse=sp.Integer(0),
        borne_haute=sp.pi / 2,
    )
    assert sp.simplify(integrale.valeur() - 3 * sp.pi / 16) == 0


def test_l_integrale_definie_montre_sa_primitive_et_ses_deux_evaluations() -> None:
    """Newton-Leibniz : le corrige doit montrer F(b) et F(a), pas leur difference."""
    integrale = IntegraleDefinie(
        integrande=2 * X,
        borne_basse=sp.Integer(1),
        borne_haute=sp.Integer(3),
    )
    en_haut, en_bas = integrale.evaluations()
    assert sp.simplify(en_haut - 9) == 0
    assert sp.simplify(en_bas - 1) == 0
    assert sp.simplify(integrale.valeur() - 8) == 0


def test_une_borne_basse_superieure_a_la_borne_haute_est_refusee() -> None:
    """L'orientation est une convention : l'inverser silencieusement trompe."""
    with pytest.raises(ValueError, match="borne basse"):
        IntegraleDefinie(
            integrande=X,
            borne_basse=sp.Integer(3),
            borne_haute=sp.Integer(1),
        )


def test_une_integrale_impropre_convergente_est_dite_convergente() -> None:
    """`int_1^oo dx/x^2 = 1` : le cas d'ecole des planches."""
    integrale = IntegraleImpropre(integrande=1 / X**2, borne_basse=sp.Integer(1))
    assert integrale.converge()
    assert sp.simplify(integrale.valeur() - 1) == 0


def test_une_integrale_impropre_divergente_est_dite_divergente() -> None:
    """`int_1^oo dx/x` diverge, et le dire vaut mieux que rendre `oo`."""
    integrale = IntegraleImpropre(integrande=1 / X, borne_basse=sp.Integer(1))
    assert not integrale.converge()


def test_la_valeur_d_une_integrale_divergente_est_refusee() -> None:
    """Rendre `oo` ferait composer un infini comme s'il etait un resultat."""
    integrale = IntegraleImpropre(integrande=1 / X, borne_basse=sp.Integer(1))
    with pytest.raises(ValueError, match="diverge"):
        integrale.valeur()


def test_l_integrale_impropre_expose_la_limite_qui_la_definit() -> None:
    """Le cours la definit par une limite : le corrige doit la montrer."""
    integrale = IntegraleImpropre(integrande=1 / X**2, borne_basse=sp.Integer(1))
    borne = integrale.borne_mobile()
    assert borne.name == "b"
    assert sp.simplify(integrale.primitive_bornee().subs(borne, 1)) == 0


@pytest.mark.parametrize(
    ("original", "attendue"),
    [
        (sp.Integer(1), 1 / IMAGE),
        (TEMPS, 1 / IMAGE**2),
        (sp.exp(-2 * TEMPS), 1 / (IMAGE + 2)),
        (sp.sin(3 * TEMPS), 3 / (IMAGE**2 + 9)),
    ],
)
def test_la_table_des_transformees_du_cours(
    original: sp.Expr,
    attendue: sp.Expr,
) -> None:
    """« Transformee de Laplace : exemples courants », planche du cours."""
    assert sp.simplify(Transformee.de(original).image - attendue) == 0


def test_la_variable_image_est_celle_du_cours() -> None:
    """Les planches notent `p`. Ecrire `s` obligerait a traduire chaque ligne."""
    assert Transformee.de(sp.Integer(1)).image.free_symbols == {IMAGE}


def test_la_transformee_de_la_derivee_fait_apparaitre_la_condition_initiale() -> None:
    """`L{y'} = p Y(p) - y(0)` : c'est ce qui rend la methode utile."""
    initiale = sp.Symbol("y_0", real=True)
    image_de_y = sp.Symbol("Y", real=True)
    attendu = IMAGE * image_de_y - initiale
    obtenu = Transformee.de_la_derivee(image=image_de_y, valeur_initiale=initiale)
    assert sp.simplify(obtenu - attendu) == 0


def test_un_probleme_de_cauchy_se_resout_par_laplace_et_se_verifie() -> None:
    """`y' + 2y = 0`, `y(0) = 3` : solution `3 e^{-2t}`, verifiee par derivation."""
    probleme = ProblemeDeCauchy(
        coefficient=sp.Integer(2),
        second_membre=sp.Integer(0),
        valeur_initiale=sp.Integer(3),
    )
    assert sp.simplify(probleme.solution() - 3 * sp.exp(-2 * TEMPS)) == 0
    assert probleme.solution_verifiee()


def test_l_equation_algebrique_en_p_est_exposee() -> None:
    """C'est l'etape que la methode apporte : l'EDO devient une equation."""
    probleme = ProblemeDeCauchy(
        coefficient=sp.Integer(2),
        second_membre=sp.Integer(0),
        valeur_initiale=sp.Integer(3),
    )
    equation = probleme.equation_en_image()
    assert equation.has(IMAGE)
    assert sp.simplify(sp.solve(equation, probleme.image)[0] - 3 / (IMAGE + 2)) == 0


def test_un_echelon_retarde_survit_a_la_reduction_de_l_echelon() -> None:
    """`Heaviside(t)` vaut un sur le domaine ; `Heaviside(t - a)` non.

    Le premier ne dit rien que la transformee unilaterale ne dise deja ;
    le second dit a quel instant la sollicitation commence. Les reduire
    tous les deux effacerait une information de l'enonce.
    """
    retard = sp.Symbol("a", positive=True)
    expression = sp.exp(-TEMPS) * sp.Heaviside(TEMPS - retard)
    assert expression.subs(sp.Heaviside(TEMPS), 1).has(sp.Heaviside)
