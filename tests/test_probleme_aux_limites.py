"""Le probleme aux limites du second ordre : `y'' = f(x, y, y')`, `y(a)`, `y(b)`.

Source : `Mathematique/SLIDES/Slides_02_Equations_differentielles.pdf`,
planche 13 : `y'' + 2 x y = 1`, `y(0) = y(1) = 0`, nomme non resolu. Deux
cas a solution connue servent d'oracle (`y'' = 2` -> `x^2 - x` ; `y'' + y = 0`
sur `[0, pi/2]` -> `sin x`) ; le cas de la planche est recoupe par la methode
de tir, un algorithme independant de `solve_bvp`.
"""

import numpy as np
import pytest
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from polytech.maths.domain.differentielles.aux_limites import (
    Bords,
    resoudre_aux_limites,
)

POINTS = np.linspace(0.0, 1.0, 11)


def test_y_seconde_egale_deux_donne_x_carre_moins_x() -> None:
    solution = resoudre_aux_limites(
        lambda x, _y, _p: np.full_like(x, 2.0), Bords(0, 1, 0, 0)
    )
    assert np.allclose(solution(POINTS), POINTS**2 - POINTS, atol=1e-6)


def test_y_seconde_plus_y_donne_le_sinus() -> None:
    bords = Bords(0.0, np.pi / 2, 0.0, 1.0)
    solution = resoudre_aux_limites(lambda _x, y, _p: -y, bords)
    x = np.linspace(0.0, np.pi / 2, 9)
    assert np.allclose(solution(x), np.sin(x), atol=1e-6)


def _tir(pente: float) -> float:
    """`y(1)` quand `y(0) = 0` et `y'(0) = pente`, pour `y'' = 1 - 2 x y`."""
    sortie = solve_ivp(
        lambda x, v: [v[1], 1 - 2 * x * v[0]],
        (0.0, 1.0),
        [0.0, pente],
        rtol=1e-10,
        atol=1e-12,
    )
    return float(sortie.y[0, -1])


def test_le_probleme_de_la_planche_13_est_recoupe_par_la_methode_de_tir() -> None:
    pente = brentq(_tir, -5.0, 5.0)
    attendu = solve_ivp(
        lambda x, v: [v[1], 1 - 2 * x * v[0]],
        (0.0, 1.0),
        [0.0, pente],
        t_eval=POINTS,
        rtol=1e-10,
        atol=1e-12,
    ).y[0]
    solution = resoudre_aux_limites(lambda x, y, _p: 1 - 2 * x * y, Bords(0, 1, 0, 0))
    assert np.allclose(solution(POINTS), attendu, atol=1e-6)


def test_les_valeurs_aux_bords_sont_respectees() -> None:
    solution = resoudre_aux_limites(lambda x, y, _p: 1 - 2 * x * y, Bords(0, 1, 0, 0))
    assert solution(np.array([0.0, 1.0])) == pytest.approx([0.0, 0.0], abs=1e-9)


def test_un_probleme_sans_solution_est_refuse() -> None:
    """`y'' + y = 0`, `y(0) = 0`, `y(pi) = 1` : le sinus ne peut valoir 1 en pi."""
    bords = Bords(0.0, np.pi, 0.0, 1.0)
    with pytest.raises(ValueError, match="pas de solution"):
        resoudre_aux_limites(lambda _x, y, _p: -y, bords)


def test_un_intervalle_vide_est_refuse() -> None:
    with pytest.raises(ValueError, match="a < b"):
        resoudre_aux_limites(lambda _x, y, _p: -y, Bords(1, 1, 0, 0))


def test_une_valeur_non_nulle_au_bord_gauche_est_respectee() -> None:
    """`y'' = 0`, `y(0) = 1`, `y(1) = 3` : la droite `1 + 2 x`."""
    solution = resoudre_aux_limites(
        lambda x, _y, _p: np.zeros_like(x), Bords(0, 1, 1, 3)
    )
    assert np.allclose(solution(POINTS), 1 + 2 * POINTS, atol=1e-8)
