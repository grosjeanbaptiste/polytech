"""La moyenne logarithmique et le rendement de la fiche, refaits par `polytech/`."""

from __future__ import annotations

import math

import sympy as sp
from scipy.optimize import brentq

from polytech.maths.domain.comparaison import egales, proches_relativement
from polytech.thermique.domain.double_vitrage import (
    Vitrage,
)
from polytech.thermique.domain.echangeurs import (
    Echangeur,
    flux_maximal_echangeable,
    moyenne_logarithmique,
)
from polytech.thermique.domain.echangeurs_a_courants_paralleles import Sens
from polytech.thermique.domain.rayonnement.corps_noir import emittance
from polytech.thermique.domain.rendement_d_echangeur import (
    rendement,
    sortie_du_fluide_chaud,
)

SIGMA_T = emittance(1.0)
VITRAGE = Vitrage(epaisseur=0.003, conductivite=0.7, echanges=(23.0, 8.0))


def test_la_moyenne_logarithmique_est_la_moyenne_d_un_ecart_exponentiel() -> None:
    """`dt_log = (dt_A - dt_B)/ln(dt_A/dt_B)` : moyenne de `dt_A e^(-k x)`."""
    da, k, x = (
        sp.Symbol("da", positive=True),
        sp.Symbol("k", positive=True),
        sp.Symbol("x"),
    )
    moyenne = sp.integrate(da * sp.exp(-k * x), (x, 0, 1))
    db = da * sp.exp(-k)
    fiche = (da - db) / sp.log(da / db)
    assert egales(moyenne, fiche)
    assert proches_relativement(
        moyenne_logarithmique(60.0, 20.0), 40.0 / math.log(3), 1e-12
    )
    assert proches_relativement(
        Echangeur(500.0).flux(60.0, 20.0), 500.0 * 40.0 / math.log(3), 1e-12
    )


def test_l_efficacite_d_une_paroi_isotherme_est_un_moins_exp_moins_nut() -> None:
    """`(t_S - t_E)/(t_p - t_E) = 1 - exp(-K_gl S/(q c))`, retrouve par la `dt_log`."""
    ks, qc, t_p, t_e = 300.0, 500.0, 100.0, 20.0

    def bilan(t_s: float) -> float:
        return qc * (t_s - t_e) - Echangeur(ks).flux(t_p - t_e, t_p - t_s)

    sortie = brentq(bilan, t_e + 1e-6, t_p - 1e-6)
    attendue = 1 - math.exp(-ks / qc)
    assert proches_relativement((sortie - t_e) / (t_p - t_e), attendue, 1e-9)
    for sens in Sens:
        assert proches_relativement(rendement(sens, ks / qc, 0.0), attendue, 1e-12)


def test_le_rendement_ne_change_pas_avec_la_temperature_d_entree() -> None:
    """`eps = Phi / ((q c)_min (t_1E - t_2E))` : le meme pour toute entree."""
    eps = rendement(Sens.CONTRECOURANT, 600.0 / 400.0, 400.0 / 800.0)
    for chaude, froide in ((90.0, 10.0), (190.0, 110.0)):
        sortie = sortie_du_fluide_chaud(chaude, froide, eps)
        flux = 400.0 * (chaude - sortie)
        maximal = flux_maximal_echangeable(400.0, 800.0, chaude, froide)
        assert proches_relativement(flux / maximal, eps, 1e-12)
