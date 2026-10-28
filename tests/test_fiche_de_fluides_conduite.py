"""Poiseuille, Blasius et le regime : la fiche de fluides, refaite.

Les trois articles de conduite (`regimes_laminaire_et_turbulent`,
`ecoulement_de_poiseuille_et_pertes_de_charge`, et le cas Navier-Stokes
de 2025) sont recalcules par `polytech.fluides.domain.conduite`.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.fluides.domain.conduite import (
    coefficient_de_blasius,
    coefficient_laminaire,
    debit_de_poiseuille,
    perte_de_charge_de_poiseuille,
    perte_de_charge_reguliere,
    profil_de_poiseuille,
)
from polytech.fluides.domain.fondamentales.continuite import vitesse_dans_la_section
from polytech.fluides.domain.proprietes import viscosite_cinematique
from polytech.fluides.domain.regime_de_reynolds import (
    REYNOLDS_LAMINAIRE,
    REYNOLDS_TURBULENT,
    RegimeDeReynolds,
    nombre_de_reynolds,
    regime_en_conduite,
)
from polytech.maths.domain.comparaison import (
    egales,
    egales_arrondies,
    proches_relativement,
)

R_, MU, G, L, RHO = sp.symbols("R mu G L rho", positive=True)
RAYON_COURANT = sp.Symbol("r", positive=True)
DEBIT, RAYON, LONGUEUR, VISCOSITE = 1e-5, 0.01, 2.0, 0.1
EAU = 1000.0


def test_le_reynolds_est_rho_u_d_sur_mu_et_u_d_sur_nu() -> None:
    nu = viscosite_cinematique(1e-3, EAU)
    assert proches_relativement(
        nombre_de_reynolds(0.6, 0.15, nu), EAU * 0.6 * 0.15 / 1e-3, 1e-12
    )


def test_les_seuils_du_regime_sont_2100_et_4000() -> None:
    assert (REYNOLDS_LAMINAIRE, REYNOLDS_TURBULENT) == (2100.0, 4000.0)
    assert regime_en_conduite(2099.0) is RegimeDeReynolds.LAMINAIRE
    assert regime_en_conduite(4001.0) is RegimeDeReynolds.TURBULENT


def test_toboggan_re_9e4_est_turbulent_donc_blasius_et_non_64_sur_re() -> None:
    vitesse = vitesse_dans_la_section(10.6e-3, math.pi * 0.15**2 / 4)
    reynolds = nombre_de_reynolds(vitesse, 0.15, 1e-3 / EAU)
    assert proches_relativement(reynolds, 9.0e4, 0.01)
    assert regime_en_conduite(reynolds) is RegimeDeReynolds.TURBULENT
    assert egales_arrondies(coefficient_de_blasius(reynolds), 0.018, 3)
    with pytest.raises(ValueError, match="laminaire"):
        coefficient_laminaire(reynolds)


def test_le_profil_est_la_solution_de_navier_stokes_reduite() -> None:
    """`(mu / r) (r u')' = -G`, `u(R) = 0`, `u` fini sur l'axe."""
    u = profil_de_poiseuille(RAYON_COURANT, R_, MU, G)
    assert egales(
        MU
        / RAYON_COURANT
        * sp.diff(RAYON_COURANT * sp.diff(u, RAYON_COURANT), RAYON_COURANT),
        -G,
    )
    assert egales(u.subs(RAYON_COURANT, R_), 0)
    assert egales(u.subs(RAYON_COURANT, 0), G * R_**2 / (4 * MU))


def test_le_debit_est_l_integrale_du_profil_sur_la_section() -> None:
    u = profil_de_poiseuille(RAYON_COURANT, R_, MU, G)
    integrale = sp.integrate(2 * sp.pi * RAYON_COURANT * u, (RAYON_COURANT, 0, R_))
    assert egales(integrale, debit_de_poiseuille(R_, MU, G))
    assert egales(debit_de_poiseuille(R_, MU, G), sp.pi * R_**4 * G / (8 * MU))


def test_la_perte_de_charge_est_g_l_et_8_mu_l_q_sur_pi_r4() -> None:
    gradient = sp.solve(sp.Eq(debit_de_poiseuille(R_, MU, G), sp.Symbol("Q")), G)[0]
    valeur = (gradient * L).subs(
        {R_: RAYON, MU: VISCOSITE, L: LONGUEUR, sp.Symbol("Q"): DEBIT}
    )
    perte = perte_de_charge_de_poiseuille(DEBIT, RAYON, LONGUEUR, VISCOSITE)
    assert proches_relativement(perte, float(valeur), 1e-12)
    assert proches_relativement(
        perte, 8 * VISCOSITE * LONGUEUR * DEBIT / (math.pi * RAYON**4), 1e-12
    )


def test_darcy_weisbach_avec_64_sur_re_redonne_poiseuille() -> None:
    vitesse = vitesse_dans_la_section(DEBIT, math.pi * RAYON**2)
    reynolds = nombre_de_reynolds(vitesse, 2 * RAYON, VISCOSITE / EAU)
    assert reynolds < REYNOLDS_LAMINAIRE
    darcy = perte_de_charge_reguliere(
        coefficient_laminaire(reynolds), LONGUEUR, 2 * RAYON, vitesse, EAU
    )
    assert proches_relativement(
        darcy, perte_de_charge_de_poiseuille(DEBIT, RAYON, LONGUEUR, VISCOSITE), 1e-12
    )


def test_janvier_2025_1_viscosite_divisee_par_deux_a_debit_constant() -> None:
    avant = perte_de_charge_de_poiseuille(DEBIT, RAYON, LONGUEUR, VISCOSITE)
    apres = perte_de_charge_de_poiseuille(DEBIT, RAYON, LONGUEUR, VISCOSITE / 2)
    assert proches_relativement(apres, avant / 2, 1e-12)


def test_a_debit_constant_le_profil_reste_le_meme_parabole() -> None:
    """`G / (4 mu) = 2 Q / (pi R^4)` : ni `G` ni `mu` n'y figurent seuls."""
    debit = sp.Symbol("Q", positive=True)

    def profil(viscosite: sp.Expr) -> sp.Expr:
        gradient = sp.solve(sp.Eq(debit_de_poiseuille(R_, viscosite, G), debit), G)[0]
        return profil_de_poiseuille(RAYON_COURANT, R_, viscosite, gradient)

    assert egales(profil(MU), profil(MU / 2))
    assert sp.degree(sp.expand(profil(MU)), RAYON_COURANT) == 2


def test_la_figure_de_poiseuille_est_proportionnelle_au_profil_de_la_bibliotheque() -> (
    None
):
    """TikZ : `2,6 (1 - (y - 1)^2)` pour `y` de 0 a 2, soit `r = y - 1`, `R = 1`."""
    u = profil_de_poiseuille(RAYON_COURANT, sp.Integer(1), sp.Integer(1), sp.Integer(1))
    figure = 2.6 * (1 - RAYON_COURANT**2)
    assert egales(figure / u, 2.6 * 4)
