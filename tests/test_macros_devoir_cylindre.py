"""Les macros du devoir cote : chaque nombre des deux rapports en sort."""

from __future__ import annotations

import math

import pytest

from polytech.fluides.application import devoir_cylindre_cas as cas
from polytech.fluides.application import devoir_cylindre_fluides as dcf
from polytech.fluides.application import macros_devoir_cylindre as mdc

MACROS = mdc.macros()


@pytest.mark.parametrize("cle", ["ChemineeFour", "ChemineeUsine", "Serre"])
def test_chaque_cas_publie_ses_grandeurs(cle: str) -> None:
    point = cas.cas(cle).point
    assert float(MACROS[f"{cle}Reynolds"]) == pytest.approx(point.reynolds, rel=0.03)
    assert MACROS[f"{cle}Regime"] == point.regime.nom
    assert float(MACROS[f"{cle}Mach"]) == pytest.approx(point.mach, abs=1e-3)


def test_l_epaisseur_de_couche_limite_au_sommet() -> None:
    """Blasius, `delta = 5 x / racine(Re_x)`, avec `x = pi R / 2`."""
    point = cas.cas("ChemineeFour").point
    x = math.pi * point.diametre / 4
    re_x = point.vitesse * x / point.fluide.viscosite_cinematique
    attendu = 100 * 5 * x / math.sqrt(re_x) / point.diametre
    assert float(MACROS["ChemineeFourCoucheLimitePourcent"]) == pytest.approx(
        attendu, rel=0.01
    )


def test_les_pressions_de_la_cheminee_de_four_en_pascals() -> None:
    """Au point d'arret `rho U^2 / 2`, au sommet `-3 rho U^2 / 2`."""
    point = cas.cas("ChemineeFour").point
    dynamique = point.fluide.masse_volumique * point.vitesse**2 / 2
    assert float(MACROS["ChemineeFourPressionArret"]) == pytest.approx(
        dynamique, rel=1e-3
    )
    assert float(MACROS["ChemineeFourDepressionSommet"]) == pytest.approx(
        -3 * dynamique, rel=1e-3
    )


def test_les_deux_sources_de_l_air_et_leur_ecart() -> None:
    ecart = 100 * abs(dcf.air_a(20.0).viscosite / dcf.AIR.viscosite - 1)
    assert float(MACROS["EcartSourcesAirPourcent"]) == pytest.approx(ecart, abs=0.05)


def test_le_rapport_des_viscosites_cinematiques() -> None:
    attendu = dcf.AIR.viscosite_cinematique / dcf.EAU.viscosite_cinematique
    assert float(MACROS["RapportViscositesCinematiques"]) == pytest.approx(
        attendu, rel=0.01
    )


def test_l_ecart_de_c_p_entre_les_cas_est_celui_de_l_arrondi() -> None:
    assert float(MACROS["CpEcartEntreLesCas"]) < 1e-12


def test_la_trainee_par_quadrature_est_nulle_a_l_arrondi() -> None:
    assert abs(mdc.trainee_par_quadrature()) < 1e-12


def test_aucune_macro_n_est_vide() -> None:
    assert all(MACROS.values())


def test_l_eau_n_a_pas_de_celerite_publiee() -> None:
    """L'eau est traitee incompressible : demander sa celerite est une faute."""
    with pytest.raises(ValueError, match="aucune celerite, le fluide est traite"):
        mdc.celerite(dcf.EAU)
