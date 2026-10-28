"""Les circuits de la fiche d'electromagnetisme, refaits par `polytech/`.

Complete `test_fiche_d_electromagnetisme.py` : Thevenin et Norton, la
fermeture, les substituts complexes, les inductances couplees, les puissances.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.circuits.equivalents import (
    courant_de_court_circuit,
    resistance_vue_des_bornes,
    tension_a_vide,
)
from polytech.electromagnetisme.domain.circuits.fermeture import (
    charge_rc,
    courant_rl,
    transformee_du_courant_rl,
)
from polytech.electromagnetisme.domain.circuits.reseau import (
    Element,
    Inductance,
    Reseau,
    Resistance,
    SourceDeTension,
)
from polytech.electromagnetisme.domain.circuits.transitoires import (
    reponse_du_premier_ordre,
)
from polytech.electromagnetisme.domain.symboles import TEMPS
from polytech.electromagnetisme.domain.transitoire.laplace import VARIABLE
from polytech.maths.domain.comparaison import egales, proches

E, R1, R2, R4, R, L, C, W = sp.symbols("E R_1 R_2 R_4 R L C omega", positive=True)


def _reseau(*, avec_la_bobine: bool) -> Reseau:
    """`E - R1 - (R4 // L) - A`, `R2` entre `A` et `B` (la reference)."""
    chaine: list[Element] = [
        SourceDeTension("n1", "B", E),
        Resistance("n1", "n2", R1),
        Resistance("n2", "A", R4),
        Resistance("A", "B", R2),
    ]
    bobine: list[Element] = [Inductance("n2", "A")] if avec_la_bobine else []
    return Reseau((*chaine, *bobine), "B")


def test_thevenin_norton_et_la_bobine_qui_court_circuite_r4() -> None:
    """`E_T` a vide, `R_T` sources eteintes, `I_N = E_T / R_T`."""
    avec, sans = _reseau(avec_la_bobine=True), _reseau(avec_la_bobine=False)
    e_t, r_t = tension_a_vide(avec, "A", "B"), resistance_vue_des_bornes(avec, "A", "B")
    assert egales(e_t, E * R2 / (R1 + R2))
    assert egales(r_t, R1 * R2 / (R1 + R2))
    assert egales(courant_de_court_circuit(avec, "A", "B"), e_t / r_t)
    assert not egales(tension_a_vide(sans, "A", "B"), e_t)  # l'oublier fausse E_T
    assert not egales(resistance_vue_des_bornes(sans, "A", "B"), r_t)  # et R_T


def test_la_fermeture_suit_la_forme_du_premier_ordre_et_ses_transformees() -> None:
    """`x = x_inf + (x_0 - x_inf) e^(-t/tau)` ; `L(L di/dt) = L (p I - i_0)`."""
    i0, tension = sp.symbols("i_0 E", positive=True)
    courant = reponse_du_premier_ordre(i0, tension / R, L / R)
    p = VARIABLE
    transformee = sp.laplace_transform(courant, TEMPS, p, noconds=True)
    derivee = sp.laplace_transform(L * sp.diff(courant, TEMPS), TEMPS, p, noconds=True)
    assert egales(derivee, L * (p * transformee - i0))
    assert egales(
        transformee_du_courant_rl(tension, R, L), p**-1 * tension / (p * L + R)
    )
    assert egales(charge_rc(tension, R, C).subs(TEMPS, 0), 0)  # u_C(0+) = u_C(0-)


def test_u_c_en_laplace_et_les_63_pour_cent_de_la_figure() -> None:
    """`U_C(p) = I(p)/(pC) + u_C(0+)/p` ; a `t = tau`, `1 - e^(-1) = 0,632`."""
    u0, tension, p = sp.Symbol("u_0", positive=True), E, VARIABLE
    charge = charge_rc(tension, R, C, initiale=C * u0)
    tr_q = sp.laplace_transform(charge, TEMPS, p, noconds=True)
    tr_i = sp.laplace_transform(sp.diff(charge, TEMPS), TEMPS, p, noconds=True)
    assert egales(tr_q / C, tr_i / (p * C) + u0 / p)
    assert proches(float(1 - sp.exp(-1)), 0.632, 5e-4)


def test_la_tension_de_bobine_saute_mais_pas_son_courant() -> None:
    """`u_L(0+) = -R i(0)` apres basculement sur `R`, alors que `i(0+) = i(0-)`."""
    i0 = sp.Symbol("i_0", positive=True)
    courant = reponse_du_premier_ordre(i0, sp.Integer(0), L / R)
    assert egales(courant.subs(TEMPS, 0), i0)
    assert egales(L * sp.diff(courant, TEMPS).subs(TEMPS, 0), -R * i0)
    assert egales(courant_rl(E, R, L).subs(TEMPS, 0), 0)
