"""Les cas particuliers de la fiche de thermique, refaits par deux chemins.

Chaque cas de `cursus/fiches/ThermiqueFondements.toml` qui se calcule est
repris ici : par la bibliotheque quand elle le porte, puis par sympy sur
l'equation de depart, sans la formule finale.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.thermique.domain.conduction.fourier import conductivite_moyenne
from polytech.thermique.domain.rayonnement.corps_noir import longueur_d_onde_du_maximum
from polytech.thermique.domain.rendement_d_echangeur import Sens, rendement

r, lam, h, t, r1, r2, phi = sp.symbols("r lambda h t r1 r2 Phi", positive=True)


def test_mai_2022_2_le_rayon_critique_vaut_lambda_sur_h() -> None:
    """Le flux 1/(ln(r/ri)/(2 pi lambda) + 1/(2 pi r h)) est maximal en lambda/h."""
    ri = sp.Symbol("r_i", positive=True)
    resistance = sp.log(r / ri) / (2 * sp.pi * lam) + 1 / (2 * sp.pi * r * h)
    assert sp.solve(sp.diff(resistance, r), r) == [lam / h]
    assert sp.diff(resistance, r, 2).subs(r, lam / h).is_positive


def test_aout_2022_2_la_resistance_d_une_coquille_spherique() -> None:
    """Fourier en symetrie spherique : T'' + 2T'/r = 0, flux constant."""
    temperature = sp.Function("T")
    generale = sp.dsolve(
        temperature(r).diff(r, 2) + 2 * temperature(r).diff(r) / r, temperature(r)
    ).rhs
    flux = sp.simplify(-lam * 4 * sp.pi * r**2 * sp.diff(generale, r))
    assert flux.free_symbols.isdisjoint({r})  # le meme flux a tout rayon
    ecart = sp.simplify(generale.subs(r, r1) - generale.subs(r, r2))
    attendue = (1 / r1 - 1 / r2) / (4 * sp.pi * lam)
    assert sp.simplify(ecart / flux - attendue) == 0


def test_aout_2022_1_wien_separe_le_soleil_du_corps_humain() -> None:
    assert 0.45e-6 < longueur_d_onde_du_maximum(5800) < 0.55e-6
    assert 9e-6 < longueur_d_onde_du_maximum(310) < 10e-6


def test_mai_2017_2_la_conductivite_moyenne_est_exacte_si_elle_est_lineaire() -> None:
    """Pour lambda = a + b t, int lambda dt / (t1 - t2) = a + b (t1 + t2) / 2."""
    a, b, t1, t2 = sp.symbols("a b t1 t2")
    exacte = sp.integrate(a + b * t, (t, t2, t1)) / (t1 - t2)
    assert sp.simplify(exacte - conductivite_moyenne(a, b, t1, t2)) == 0


def test_aout_2016_2_une_capacite_infinie_donne_1_moins_exp_moins_nut() -> None:
    for nut in (0.3, 1.0, 2.5):
        attendue = 1 - math.exp(-nut)
        for sens in Sens:
            assert math.isclose(rendement(sens, nut, 0.0), attendue)


def test_mai_2022_1_huit_tubes_en_parallele_divisent_le_reynolds_par_huit() -> None:
    debit, d, nu, rho = sp.symbols("m D nu rho", positive=True)
    vitesse = (debit / 8) / (rho * sp.pi * d**2 / 4)
    reynolds = vitesse * d / nu
    seul = (debit / (rho * sp.pi * d**2 / 4)) * d / nu
    assert sp.simplify(reynolds / seul) == sp.Rational(1, 8)
