"""La dynamique du point, d'apres le syllabus « Notions fondamentales ».

Source : `PhysiqueGenerale/SYLLABUS/Syllabus_Notions_fondamentales_2026_2027.pdf`
(« Physique generale », UMONS, 2026-2027). Chaque oracle est une equation
du syllabus, citee par son numero ; chaque test la confronte a un second
chemin -- integrer, resoudre les lois de Newton avec sympy, verifier une
conservation -- pour que le code ne se contente pas de recopier la formule.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.mecanique.domain.dynamique.cinematique import (
    position_mrua,
    vitesse_mrua,
    vitesse_par_la_position,
)
from polytech.mecanique.domain.dynamique.collisions import collision_elastique
from polytech.mecanique.domain.dynamique.newton import (
    acceleration_centripete,
    atwood,
    plan_incline,
)

t, a, v0, x0 = sp.symbols("t a v0 x0", real=True)


def test_mrua_vitesse_eq_2_10_est_l_integrale_de_l_acceleration() -> None:
    """(2.10) v = v0 + a t, p. 17 ; second chemin : v0 + integrale de a."""
    attendu = v0 + sp.integrate(a, (t, 0, t))
    assert sp.simplify(vitesse_mrua(v0, a, t) - attendu) == 0


def test_mrua_position_eq_2_14_est_l_integrale_de_la_vitesse() -> None:
    """(2.14) x = x0 + v0 t + a t^2 / 2, p. 18."""
    s = sp.Symbol("s", real=True)
    attendu = x0 + sp.integrate(vitesse_mrua(v0, a, s), (s, 0, t))
    assert sp.simplify(position_mrua(x0, v0, a, t) - attendu) == 0


def test_mrua_eq_2_15_elimine_le_temps() -> None:
    """(2.15) v^2 = v0^2 + 2 a (x - x0), p. 18, refait par (2.10) et (2.14)."""
    v = vitesse_mrua(v0, a, t)
    x = position_mrua(x0, v0, a, t)
    assert sp.simplify(vitesse_par_la_position(v0, a, x0, x) ** 2 - v**2) == 0


def test_mrua_sans_acceleration_est_uniforme() -> None:
    """(2.16) et (2.17), p. 18 : a = 0 donne le mouvement rectiligne uniforme."""
    assert vitesse_mrua(3.0, 0.0, 5.0) == 3.0
    assert position_mrua(1.0, 3.0, 0.0, 5.0) == 16.0


def test_vitesse_par_la_position_refuse_un_point_jamais_atteint() -> None:
    """v^2 < 0 : la particule n'atteint jamais cette position."""
    with pytest.raises(ValueError, match="jamais atteinte"):
        vitesse_par_la_position(1.0, -1.0, 0.0, 10.0)
    # Juste sous zero : v^2 = -0,5 n'est pas une vitesse.
    with pytest.raises(ValueError, match="jamais atteinte"):
        vitesse_par_la_position(0.0, -1.0, 0.0, 0.25)


def _atwood_par_newton() -> tuple[sp.Expr, sp.Expr]:
    """Les deux lois de Newton de la machine, resolues pour a et T."""
    m1, m2, g, acc, tension = sp.symbols("m1 m2 g a T", positive=True)
    solution = sp.solve(
        [tension - m1 * g - m1 * acc, tension - m2 * g + m2 * acc], [acc, tension]
    )
    return solution[acc], solution[tension]


def test_atwood_eq_4_16_et_4_17_par_les_lois_de_newton() -> None:
    """(4.16) a = (m2 - m1) g / (m1 + m2), (4.17) T = 2 m1 m2 g / (m1 + m2), p. 43."""
    m1, m2, g = sp.symbols("m1 m2 g", positive=True)
    acc, tension = atwood(m1, m2, g)
    a_newton, t_newton = _atwood_par_newton()
    assert sp.simplify(acc - a_newton) == 0
    assert sp.simplify(tension - t_newton) == 0


def test_atwood_masses_egales_est_l_equilibre() -> None:
    """P. 43 : deux masses egales, l'acceleration est nulle."""
    assert atwood(2.0, 2.0, 9.81)[0] == 0.0


def test_plan_incline_eq_4_21_4_22_4_25() -> None:
    """(4.21) T = m g sin, (4.22) N = m g cos, (4.25) a = -g sin, p. 44."""
    m, g, theta = sp.symbols("m g theta", positive=True)
    plan = plan_incline(m, g, theta)
    assert sp.simplify(plan.tension - m * g * sp.sin(theta)) == 0
    assert sp.simplify(plan.normale - m * g * sp.cos(theta)) == 0
    # Second chemin : le poids projete ; sans corde, N - m g cos = 0 et m a = -m g sin.
    poids = sp.Matrix([-m * g * sp.sin(theta), -m * g * sp.cos(theta)])
    assert sp.simplify(plan.acceleration - poids[0] / m) == 0


def test_plan_incline_horizontal_ne_glisse_pas() -> None:
    assert plan_incline(1.0, 9.81, 0.0).acceleration == pytest.approx(0.0)


def test_acceleration_centripete_eq_4_26_par_la_derivee_seconde() -> None:
    """(4.26) a = v^2 / r, p. 44 ; second chemin : |d^2/dt^2 (r cos wt, r sin wt)|."""
    r, w, tt = sp.symbols("r w t", positive=True)
    position = sp.Matrix([r * sp.cos(w * tt), r * sp.sin(w * tt)])
    norme = sp.sqrt(sum(c**2 for c in position.diff(tt, 2)))
    assert sp.simplify(acceleration_centripete(r * w, r) - norme) == 0


def test_acceleration_centripete_refuse_un_rayon_nul() -> None:
    with pytest.raises(ValueError, match="rayon"):
        acceleration_centripete(1.0, 0.0)


@pytest.mark.parametrize(
    ("m1", "v1", "m2", "v2"),
    [(1.0, 3.0, 2.0, -1.0), (5.0, 2.0, 0.1, 0.0), (0.2, 4.0, 7.0, 1.5)],
)
def test_collision_elastique_conserve_quantite_de_mouvement_et_energie(
    m1: float, v1: float, m2: float, v2: float
) -> None:
    """(6.42) et (6.43), p. 74 : ce que (6.48) et (6.49) doivent respecter."""
    v1f, v2f = collision_elastique(m1, v1, m2, v2)
    assert m1 * v1 + m2 * v2 == pytest.approx(m1 * v1f + m2 * v2f)
    assert m1 * v1**2 + m2 * v2**2 == pytest.approx(m1 * v1f**2 + m2 * v2f**2)


def test_collision_de_masses_egales_echange_les_vitesses() -> None:
    """(6.50), p. 75."""
    assert collision_elastique(2.0, 3.0, 2.0, -1.0) == (-1.0, 3.0)


def test_collision_sur_une_masse_tres_lourde_rebondit() -> None:
    """(6.54), p. 75 : m2 >> m1 au repos, v1f ~ -v1i et v2f ~ 0."""
    v1f, v2f = collision_elastique(1e-6, 4.0, 1.0, 0.0)
    assert v1f == pytest.approx(-4.0, rel=1e-5)
    assert v2f == pytest.approx(0.0, abs=1e-4)


def test_collision_refuse_une_masse_nulle() -> None:
    with pytest.raises(ValueError, match="masse"):
        collision_elastique(0.0, 1.0, 1.0, 0.0)


def test_mrua_chiffre() -> None:
    """Un cas chiffre : x0 = 0, v0 = 2 m/s, a = 3 m/s^2, t = 4 s."""
    assert position_mrua(0.0, 2.0, 3.0, 4.0) == pytest.approx(32.0)
    assert vitesse_par_la_position(2.0, 3.0, 0.0, 32.0) == pytest.approx(
        math.sqrt(2.0**2 + 2 * 3.0 * 32.0)
    )
