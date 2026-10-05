"""Le syllabus de thermique, tome 2 : l'analyse dimensionnelle de la convection.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_2_2013.pdf`, pages 21
a 31 (premiere partie, chapitre 2), lues sur les pages rendues.

Les oracles : les equations aux dimensions de la page 24, les nombres
(2.1) a (2.9), et le theoreme des pi lui-meme, applique par le contexte
des fluides aux sept variables que le syllabus enumere.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.chemistry.domain.gaz import AIR_SEC, masse_volumique_du_gaz_parfait
from polytech.fluides.domain.analyse_dimensionnelle import nombre_de_groupes
from polytech.fluides.domain.proprietes import (
    coefficient_de_dilatation_du_gaz_parfait,
)
from polytech.fluides.domain.variables_repetees import (
    dimensions_du_produit,
    groupes_pi,
)
from polytech.thermique.domain.convection.forcee import (
    coefficient_depuis_le_nusselt,
    prandtl,
    reynolds,
)
from polytech.thermique.domain.convection.naturelle import grashof, rayleigh
from polytech.thermique.domain.convection.similitude import (
    CONVECTION_FORCEE,
    CONVECTION_NATURELLE,
    masse_volumique_dilatee,
    nusselt,
    peclet,
    prandtl_des_proprietes,
    stanton,
    temperature_de_film,
    temperature_de_melange,
    viscosite_pres_du_melange,
)

REPETES = ("l", "nu", "theta_o", "lambda")


def test_sept_variables_et_quatre_unites_font_trois_nombres() -> None:
    """Page 24 : `phi = f(l, u_o, nu, theta_o, a, lambda)`, en quatre unites."""
    assert len(CONVECTION_FORCEE) == 7
    unites = {
        unite for dimensions in CONVECTION_FORCEE.values() for unite in dimensions
    }
    assert unites == {"L", "T", "Theta", "Q"}
    assert nombre_de_groupes(7, 4) == 3
    assert CONVECTION_FORCEE["phi"] == {"Q": 1, "L": -2, "T": -1}
    assert CONVECTION_FORCEE["lambda"] == {"Q": 1, "L": -1, "T": -1, "Theta": -1}


def test_le_theoreme_des_pi_rend_nusselt_reynolds_et_prandtl() -> None:
    """(2.1) a (2.4) : `Nu = phi l / (lambda theta_o)`, `Re = u_o l / nu`, `Pr`."""
    nu, inverse_de_pr, re = groupes_pi("phi", REPETES, CONVECTION_FORCEE)
    assert nu == {"phi": 1, "l": 1, "theta_o": -1, "lambda": -1}
    assert re == {"u_o": 1, "l": 1, "nu": -1}
    assert inverse_de_pr == {"a": 1, "nu": -1}
    for groupe in (nu, re, inverse_de_pr):
        assert dimensions_du_produit(groupe, CONVECTION_FORCEE) == {}


def test_en_convection_naturelle_le_grashof_remplace_le_reynolds() -> None:
    """(2.5) et (2.6) : `Gr = beta g theta_o l^3 / nu^2`, et `Nu = F(Gr, Pr)`."""
    assert "u_o" not in CONVECTION_NATURELLE
    assert CONVECTION_NATURELLE["beta_g"] == {"L": 1, "T": -2, "Theta": -1}
    nu, inverse_de_pr, gr = groupes_pi("phi", REPETES, CONVECTION_NATURELLE)
    assert nu == {"phi": 1, "l": 1, "theta_o": -1, "lambda": -1}
    assert gr == {"beta_g": 1, "theta_o": 1, "l": 3, "nu": -2}
    assert inverse_de_pr == {"a": 1, "nu": -1}
    numerique = grashof(1 / 300.0, 20.0, 0.5, 1.6e-5)
    assert numerique == pytest.approx(9.81 / 300.0 * 20.0 * 0.5**3 / 1.6e-5**2)


def test_le_nusselt_compare_a_la_conduction_pure() -> None:
    """Page 27 : `Nu = phi / (lambda theta_o / l) = K l / lambda`."""
    assert nusselt(25.0, 0.2, 0.026) == pytest.approx(25.0 * 0.2 / 0.026)
    couche_au_repos = 0.026 / 0.2
    assert nusselt(couche_au_repos, 0.2, 0.026) == pytest.approx(1.0)
    retour = coefficient_depuis_le_nusselt(nusselt(25.0, 0.2, 0.026), 0.026, 0.2)
    assert retour == pytest.approx(25.0)
    with pytest.raises(ValueError, match="conductivite nulle ou negative"):
        nusselt(25.0, 0.2, 0.0)


def test_le_prandtl_des_gaz_usuels_est_voisin_de_0_7() -> None:
    """(2.7) : `Pr = nu / a = mu c_p / lambda` ; l'air a 20 degres."""
    par_les_proprietes = prandtl_des_proprietes(1.81e-5, 1005.0, 0.0257)
    assert par_les_proprietes == pytest.approx(0.7, abs=0.02)
    masse_volumique = 1.2
    cinematique = 1.81e-5 / masse_volumique
    diffusivite = 0.0257 / (masse_volumique * 1005.0)
    assert prandtl(cinematique, diffusivite) == pytest.approx(par_les_proprietes)


def test_les_autres_variables_reduites() -> None:
    """Page 28 : `Pe = Re Pr`, `Ray = Gr Pr`, `St = Nu / (Re Pr)`."""
    re = reynolds(2.0, 0.05, 1.5e-5)
    assert peclet(re, 0.7) == pytest.approx(0.7 * 2.0 * 0.05 / 1.5e-5)
    assert rayleigh(4e8, 0.7) == pytest.approx(2.8e8)
    assert stanton(40.0, re, 0.7) == pytest.approx(40.0 / (re * 0.7))
    echange, masse_volumique, chaleur = 30.0, 1.2, 1005.0
    par_definition = echange / (masse_volumique * chaleur * 2.0)
    diffusivite = 0.026 / (masse_volumique * chaleur)
    nombre = nusselt(echange, 0.05, 0.026)
    autre = stanton(nombre, reynolds(2.0, 0.05, 1.5e-5), 1.5e-5 / diffusivite)
    assert autre == pytest.approx(par_definition)


def test_la_dilatation_cree_la_force_d_archimede() -> None:
    """Page 26 : `rho = rho_o [1 - beta (t - t_o)]`, `beta = -(1/rho) d(rho)/dT`."""
    assert masse_volumique_dilatee(1.2, 1 / 300.0, 30.0) == pytest.approx(1.08)
    beta = coefficient_de_dilatation_du_gaz_parfait(300.0)
    assert beta == pytest.approx(1 / 300.0)
    pas = 1e-3
    plus = masse_volumique_du_gaz_parfait(101325.0, 300.0 + pas, AIR_SEC)
    moins = masse_volumique_du_gaz_parfait(101325.0, 300.0 - pas, AIR_SEC)
    centre = masse_volumique_du_gaz_parfait(101325.0, 300.0, AIR_SEC)
    assert -(plus - moins) / (2 * pas) / centre == pytest.approx(beta, rel=1e-6)


def test_la_temperature_de_film() -> None:
    """(2.9) : `t_film = (t_p + t_o) / 2`, la moyenne de la couche limite."""
    assert temperature_de_film(80.0, 20.0) == 50.0


def test_la_temperature_de_melange_pondere_par_le_debit() -> None:
    """Page 30 : `t_m = (1/q_m) int t u rho dA`, et `phi = q_m c_p Dt_m`."""
    rayon = sp.Symbol("r", nonnegative=True)
    grand = sp.Symbol("R", positive=True)
    parabolique = 2 * (1 - rayon**2 / grand**2)
    uniforme = temperature_de_melange(sp.Integer(60), parabolique, rayon, grand)
    assert uniforme == 60
    chaud_au_bord = 40 + 40 * rayon**2 / grand**2
    melange = temperature_de_melange(chaud_au_bord, parabolique, rayon, grand)
    assert melange == sp.Rational(160, 3)
    surfacique = sp.integrate(chaud_au_bord * 2 * rayon, (rayon, 0, grand)) / grand**2
    assert surfacique == 60
    assert melange < surfacique


def test_la_viscosite_se_linearise_autour_du_melange() -> None:
    """Page 31 : `mu / mu_m ~ 1 + (mu_p / mu_m - 1)(t - t_m)/(t_p - t_m)`."""
    proprietes = {"melange": (40.0, 0.03), "paroi": (90.0, 0.006)}
    assert viscosite_pres_du_melange(40.0, **proprietes) == pytest.approx(0.03)
    assert viscosite_pres_du_melange(90.0, **proprietes) == pytest.approx(0.006)
    assert viscosite_pres_du_melange(65.0, **proprietes) == pytest.approx(0.018)
    with pytest.raises(ValueError, match="paroi a la temperature de melange"):
        viscosite_pres_du_melange(40.0, melange=(40.0, 0.03), paroi=(40.0, 0.006))
