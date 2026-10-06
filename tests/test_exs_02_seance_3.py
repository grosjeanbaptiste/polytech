"""Mathematique, exercices 2, seance III : systemes differentiels lineaires.

Source : `Mathematique/EXS/Exs_02_Equations_differentielles.pdf`, pages 17
a 19. Chaque reponse de la feuille est reportee dans son systeme.
"""

from __future__ import annotations

import sympy as sp

from polytech.maths.application import systemes_exs_02 as sy
from polytech.maths.domain.differentielles.plan_de_phase import Nature
from polytech.maths.domain.symboles import IMAGE as P
from polytech.maths.domain.symboles import TEMPS as T

C1, C2 = sy.C1, sy.C2


def test_iii1_les_modes_propres() -> None:
    """Page 17 : `(1, sqrt 2) e^(-t)` et `(-sqrt 2, 1) e^(-4t)`."""
    modes = dict(sy.modes_du_1())
    assert set(modes) == {-1, -4}
    assert modes[-1] == sp.Matrix([1, sp.sqrt(2)])
    assert modes[-4] == sp.Matrix([-sp.sqrt(2), 1])


def test_iii1_le_wronskien() -> None:
    """Page 17 : `3 e^(-5t)`, jamais nul ; et `-5` est la trace de `A`."""
    wronskien, par_la_trace = sy.wronskiens_du_1()
    assert sp.simplify(wronskien - 3 * sp.exp(-5 * T)) == 0
    assert sp.simplify(par_la_trace - wronskien) == 0


def test_iii1_l_origine_est_un_noeud_stable() -> None:
    nature = sy.nature_du_1()
    assert nature is Nature.NOEUD_STABLE
    assert nature.est_stable


def test_iii1_une_trajectoire_suit_le_mode_lent() -> None:
    """Quand `t` croit, `e^(-4t)` s'efface : on s'aligne sur `(1, sqrt 2)`."""
    trajectoire = sy.trajectoire_du_1(1, 1)
    depart = sp.Matrix([1 - sp.sqrt(2), sp.sqrt(2) + 1])
    assert sp.simplify(trajectoire.subs(T, 0) - depart) == sp.zeros(2, 1)
    pente = sp.limit(trajectoire[1] / trajectoire[0], T, sp.oo)
    assert pente == sp.sqrt(2)


def test_iii2_la_solution_de_la_feuille_par_les_deux_methodes() -> None:
    par_laplace, par_exponentielle = sy.solutions_du_2()
    attendue = sp.Matrix(
        [
            (C1 / 2 + C2 / 4) * sp.exp(-T) + (C1 / 2 - C2 / 4) * sp.exp(3 * T),
            (C1 + C2 / 2) * sp.exp(-T) + (-C1 + C2 / 2) * sp.exp(3 * T),
        ]
    )
    assert sp.simplify(par_laplace - attendue) == sp.zeros(2, 1)
    assert sp.simplify(par_exponentielle - attendue) == sp.zeros(2, 1)


def test_iii2_la_resolvante() -> None:
    """`(pI - A)^-1`, ce que la methode de Laplace calcule d'abord."""
    p = P
    attendue = sp.Matrix([[p - 1, -1], [-4, p - 1]]) / ((p + 1) * (p - 3))
    assert sp.simplify(sy.resolvante_du_2() - attendue) == sp.zeros(2, 2)


def test_iii3_la_solution_de_la_feuille() -> None:
    """Page 18 : un foyer, `e^(-6t)` fois une rotation."""
    solution = sy.solution_du_3()
    attendue = sp.exp(-6 * T) * sp.Matrix(
        [
            C1 * sp.cos(T) + (C2 - C1) * sp.sin(T),
            C2 * sp.cos(T) + (C2 - 2 * C1) * sp.sin(T),
        ]
    )
    assert sp.simplify(solution - attendue) == sp.zeros(2, 1)
    assert set(sy.valeurs_propres_du_3()) == {-6 + sp.I, -6 - sp.I}


def test_iii4_la_solution_de_la_feuille() -> None:
    x, y = sy.solution_du_4()
    assert sp.simplify(x - (-(T**3) / 6 - T - 1 + sp.exp(T))) == 0
    assert sp.simplify(y - (T**4 / 24 + T**2 + T + 1 - sp.exp(T))) == 0


def test_iii4_la_solution_verifie_le_systeme_et_le_repos() -> None:
    x, y = sy.solution_du_4()
    assert sp.simplify(x.diff(T, 2) + y.diff(T) + x - sp.exp(T)) == 0
    assert sp.simplify(x.diff(T) + y.diff(T, 2) - 1) == 0
    initiales = [f.subs(T, 0) for f in (x, x.diff(T), y, y.diff(T))]
    assert initiales == [0, 0, 0, 0]


def test_iii4_les_images() -> None:
    """Ce que la feuille ne montre pas : `X(p)` et `Y(p)`."""
    p = P
    image_x, image_y = sy.images_du_4()
    assert sp.simplify(image_x - (p**2 - p + 1) / (p**4 * (p - 1))) == 0
    attendue = 1 / p**5 + 2 / p**3 + 1 / p**2 + 1 / p - 1 / (p - 1)
    assert sp.simplify(image_y - attendue) == 0
    assert sp.simplify(p * image_x + p**2 * image_y - 1 / p) == 0


def test_iii5_la_solution_de_la_feuille() -> None:
    x, y = sy.solution_du_5()
    attendu_x = (
        sp.Rational(1, 15)
        - sp.exp(-T) / 2
        - sp.exp(-3 * T) / 6
        + 3 * sp.exp(-5 * T) / 5
    )
    attendu_y = sp.Rational(2, 15) + sp.exp(-3 * T) / 6 - 3 * sp.exp(-5 * T) / 10
    assert sp.simplify(x - attendu_x) == 0
    assert sp.simplify(y - attendu_y) == 0


def test_iii5_la_solution_verifie_le_systeme() -> None:
    x, y = sy.solution_du_5()
    assert sp.simplify(x.diff(T) + x + 3 * y.diff(T) + 7 * y - 1) == 0
    assert sp.simplify(-2 * x + y.diff(T) + y - sp.exp(-T)) == 0
    assert (x.subs(T, 0), y.subs(T, 0)) == (0, 0)


def test_iii5_le_regime_permanent() -> None:
    """`x -> 1/15`, `y -> 2/15` : la solution de `x + 7y = 1`, `y = 2x`."""
    assert sy.permanent_du_5() == (sp.Rational(1, 15), sp.Rational(2, 15))


def test_iii1_les_deux_directions_propres_sont_orthogonales() -> None:
    (_, lent), (_, rapide) = sy.modes_du_1()
    assert sp.simplify(lent.dot(rapide)) == 0


def test_les_natures_des_trois_origines() -> None:
    """Un noeud stable, une selle, un foyer stable."""
    assert sy.natures() == (Nature.NOEUD_STABLE, Nature.SELLE, Nature.FOYER_STABLE)


def test_iii2_les_seules_solutions_qui_reviennent_a_l_origine() -> None:
    """Le mode `e^(3t)` s'annule dans les deux composantes quand `C2 = 2 C1`."""
    condition = sy.retour_a_l_origine_du_2()
    assert condition == 2 * C1
    par_laplace, _ = sy.solutions_du_2()
    bornee = par_laplace.subs(C2, condition)
    assert sp.limit(bornee[0], T, sp.oo) == 0
    assert sp.limit(bornee[1], T, sp.oo) == 0


def test_iii1_la_trajectoire_par_un_point_donne() -> None:
    """Les constantes `c1`, `c2` se tirent de la position initiale."""
    depart = sp.Matrix([2, -1])
    trajectoire = sy.trajectoire_depuis(depart)
    assert sp.simplify(trajectoire.subs(T, 0) - depart) == sp.zeros(2, 1)
    ecart = trajectoire.diff(T) - sy.PREMIER.matrice * trajectoire
    assert sp.simplify(ecart) == sp.zeros(2, 1)
