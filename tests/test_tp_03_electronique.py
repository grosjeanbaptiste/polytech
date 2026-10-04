"""Le TP 3 d'electronique : Laplace, et quadripoles en cascade.

Chaque reponse temporelle est verifiee par un chemin qui ne passe pas
par sa construction : l'equation differentielle du circuit, qu'elle doit
annuler, et ses conditions initiales. Les parties asymptotiques sont
recoupees par les phaseurs, et le cas `omega = 0` doit redonner la
reponse indicielle. Le filtre de l'exercice 3.2 est compare a l'officiel.
"""

from __future__ import annotations

import sympy as sp

from polytech.electronique.application.laplace_tp_03 import (
    indicielle_du_2_1,
    indicielle_du_2_2,
    poles_du_2_2,
    regime_du_2_1,
    reponse_au_cosinus_du_2_1,
    resonance_sans_resistance,
    transfert_du_2_1,
    transfert_du_2_2,
)
from polytech.electronique.application.quadripoles_tp_03 import (
    filtre_numerique,
    transfert_de_la_cascade,
)
from polytech.electronique.application.symboles_tp_03 import RL, C, L, R, S, T, W


def _nul(expression: sp.Expr) -> bool:
    return bool(sp.simplify(expression) == 0)


def test_2_1_le_transfert_est_un_passe_haut() -> None:
    assert _nul(transfert_du_2_1() - S * R * C / (1 + S * R * C))


def test_2_1_l_indicielle_annule_son_equation_et_part_de_un() -> None:
    """`RC (v' - u') + v = 0` pour t > 0 : `RC v' + v = 0`, `v(0+) = 1`."""
    v = indicielle_du_2_1()
    assert _nul(R * C * sp.diff(v, T) + v)
    assert _nul(v.subs(T, 0) - 1)


def test_2_1_la_reponse_au_cosinus_annule_son_equation() -> None:
    """`RC v' + v = RC e'` avec `e = cos(omega t)`, et `v(0+) = 1`."""
    v = reponse_au_cosinus_du_2_1()
    entree = sp.cos(W * T)
    assert _nul(sp.expand_trig(R * C * sp.diff(v, T) + v - R * C * sp.diff(entree, T)))
    assert _nul(v.subs(T, 0) - 1)


def test_2_1_a_pulsation_nulle_on_retrouve_l_indicielle() -> None:
    assert _nul(reponse_au_cosinus_du_2_1().subs(W, 0) - indicielle_du_2_1())


def test_2_2_le_transfert_est_un_passe_bas_du_second_ordre() -> None:
    assert _nul(transfert_du_2_2() - 1 / (L * C * S**2 + R * C * S + 1))


def test_2_2_les_poles_annulent_le_denominateur() -> None:
    for pole in poles_du_2_2():
        assert _nul((L * C * S**2 + R * C * S + 1).subs(S, pole))


def test_2_2_l_indicielle_annule_son_equation() -> None:
    """`LC v'' + RC v' + v = 1`, `v(0) = v'(0) = 0`."""
    v = indicielle_du_2_2()
    assert _nul(L * C * sp.diff(v, T, 2) + R * C * sp.diff(v, T) + v - 1)
    assert _nul(v.subs(T, 0))
    assert _nul(sp.diff(v, T).subs(T, 0))


def test_2_2_sans_resistance_la_resonance_croit_lineairement() -> None:
    """A `omega = 1/racine(LC)`, `LC v'' + v = cos` : solution en `t sin`."""
    v = resonance_sans_resistance()
    w0 = 1 / sp.sqrt(L * C)
    assert _nul(L * C * sp.diff(v, T, 2) + v - sp.cos(w0 * T))
    assert _nul(v.subs(T, 0))


def test_3_2_le_transfert_de_la_cascade() -> None:
    attendu = RL / ((1 + S * R * C) * RL + S * L + R * (1 + S**2 * L * C))
    assert _nul(transfert_de_la_cascade() - attendu)


def test_3_2_le_filtre_est_un_butterworth_coupant_a_deux_cent_mille() -> None:
    module_carre, a_l_origine, coupure = filtre_numerique()
    assert a_l_origine == sp.Rational(1, 4)
    assert coupure == 200_000
    assert _nul(module_carre.subs(W, coupure) / a_l_origine - sp.Rational(1, 2))


def test_3_2_le_module_a_exactement_la_forme_de_butterworth() -> None:
    """`|H|^2 = |H(0)|^2 / (1 + (omega / omega_c)^4)`, sans approximation."""
    module_carre, a_l_origine, coupure = filtre_numerique()
    assert _nul(module_carre - a_l_origine / (1 + (W / coupure) ** 4))


def test_2_1_la_reponse_tend_vers_le_regime_des_phaseurs() -> None:
    """Ce qui reste, une fois le regime ote, s'eteint en `e^{-t/RC}`."""
    reste = reponse_au_cosinus_du_2_1() - regime_du_2_1()
    assert _nul(sp.limit(reste, T, sp.oo))
