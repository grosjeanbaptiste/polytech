"""Les cas particuliers de la fiche de mecanique des fluides, refaits.

Le chapitre V du cours (ecoulements en conduite) manque au depot : Poiseuille
se refait ici depuis Navier-Stokes, resolu par sympy, et recoupe par le
debit qu'on integre sur la section. Les cas chiffres de l'examen (similitude,
parapente, toboggan) le sont deja dans `test_examen_de_mecanique_des_fluides.py`.
"""

from __future__ import annotations

import sympy as sp

y, h, mu, g = sp.symbols("y h mu G", positive=True)


def _poiseuille_plan(viscosite: sp.Expr) -> sp.Expr:
    """`mu u'' = -G` entre deux plaques fixes en y = 0 et y = h."""
    u = sp.Function("u")
    solution = sp.dsolve(
        sp.Eq(viscosite * u(y).diff(y, 2), -g), u(y), ics={u(0): 0, u(h): 0}
    )
    return solution.rhs


def test_a_debit_constant_la_perte_de_charge_suit_la_viscosite() -> None:
    """Janvier 2025, question 1 : viscosite divisee par deux, gradient aussi."""
    debit = sp.integrate(_poiseuille_plan(mu), (y, 0, h))
    gradient = sp.solve(sp.Eq(debit, sp.Symbol("Q")), g)[0]
    moitie = sp.solve(
        sp.Eq(sp.integrate(_poiseuille_plan(mu / 2), (y, 0, h)), sp.Symbol("Q")), g
    )[0]
    assert sp.simplify(moitie / gradient) == sp.Rational(1, 2)


def test_le_profil_laminaire_est_une_parabole() -> None:
    profil = _poiseuille_plan(mu)
    assert sp.degree(sp.expand(profil), y) == 2
    assert sp.solve(sp.diff(profil, y), y) == [h / 2]


def test_entre_deux_plaques_la_contrainte_est_lineaire_et_nulle_au_centre() -> None:
    """Janvier 2025, question 4 : tau = mu u' , nulle a mi-hauteur."""
    contrainte = mu * sp.diff(_poiseuille_plan(mu), y)
    assert sp.degree(sp.expand(contrainte), y) == 1
    assert sp.simplify(contrainte.subs(y, h / 2)) == 0
