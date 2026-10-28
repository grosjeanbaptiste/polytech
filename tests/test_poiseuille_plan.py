"""Poiseuille entre deux plaques fixes : `mu u'' = -G`, `u(0) = u(h) = 0`."""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.poiseuille_plan import debit_plan, profil_plan
from polytech.maths.domain.comparaison import egales

Y, H, MU, G = sp.symbols("y h mu G", positive=True)


def test_le_profil_verifie_l_equation_reduite_et_l_adherence() -> None:
    u = profil_plan(Y, H, MU, G)
    assert egales(MU * sp.diff(u, Y, 2), -G)
    assert egales(u.subs(Y, 0), 0)
    assert egales(u.subs(Y, H), 0)


def test_le_profil_est_maximal_a_mi_hauteur() -> None:
    u = profil_plan(Y, H, MU, G)
    assert sp.solve(sp.diff(u, Y), Y) == [H / 2]


def test_le_debit_par_unite_de_largeur_est_l_integrale_du_profil() -> None:
    integrale = sp.integrate(profil_plan(Y, H, MU, G), (Y, 0, H))
    assert egales(integrale, debit_plan(H, MU, G))
    assert egales(debit_plan(H, MU, G), G * H**3 / (12 * MU))


def test_la_contrainte_est_lineaire_et_nulle_au_centre() -> None:
    contrainte = MU * sp.diff(profil_plan(Y, H, MU, G), Y)
    assert egales(contrainte.subs(Y, H / 2), 0)
    assert egales(sp.diff(contrainte, Y, 2), 0)
