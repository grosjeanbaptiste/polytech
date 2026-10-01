"""Les gardes des modules thermiques et electromagnetiques ajoutes.

L'audit du depot exige qu'aucun `raise` ne reste jamais execute. Celles-ci
protegent de geometries impossibles -- une paroi d'epaisseur nulle, une
surface d'aire nulle, un cycle d'hysteresis de volume negatif -- qui
rendraient des resistances infinies ou nulles que le document composerait
comme des resultats.
"""

from __future__ import annotations

import pytest

from polytech.electromagnetisme.domain.magnetostatique.circuits_magnetiques import (
    pertes_par_hysteresis,
)
from polytech.electromagnetisme.domain.magnetostatique.couplage import (
    inductance_propre_d_un_solenoide,
)
from polytech.thermique.domain.convection.naturelle import (
    coefficient_global_de_transmission,
    resistance_de_paroi_cylindrique,
    resistance_de_paroi_plane,
    resistance_de_surface,
)
from polytech.thermique.domain.echangeurs import capacite_minimale
from polytech.thermique.domain.rayonnement.corps_noir import fraction_emise_sous
from polytech.thermique.domain.rayonnement.surfaces_grises import (
    flux_entre_deux_plans_gris,
)


def test_une_paroi_plane_d_epaisseur_nulle_est_refusee() -> None:
    """Sa resistance serait nulle : la paroi disparaitrait du schema."""
    with pytest.raises(ValueError, match="paroi plane invalide"):
        resistance_de_paroi_plane(0.0, 1.0, 1.0)


def test_une_paroi_plane_de_conductivite_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="paroi plane invalide"):
        resistance_de_paroi_plane(0.01, 0.0, 1.0)


def test_une_paroi_cylindrique_sans_longueur_est_refusee() -> None:
    with pytest.raises(ValueError, match="non positive"):
        resistance_de_paroi_cylindrique(0.01, 0.02, 1.0, 0.0)


def test_une_resistance_de_surface_sans_aire_est_refusee() -> None:
    with pytest.raises(ValueError, match="non positif"):
        resistance_de_surface(8.0, 0.0)


def test_un_coefficient_global_sur_une_aire_nulle_est_refuse() -> None:
    with pytest.raises(ValueError, match="aire nulle ou negative"):
        coefficient_global_de_transmission((1.0,), 0.0)


def test_une_capacite_calorifique_non_positive_est_refusee() -> None:
    with pytest.raises(ValueError, match="capacite calorifique non positive"):
        capacite_minimale(0.0, 1000.0)


def test_une_temperature_nulle_n_a_pas_de_fraction_emise() -> None:
    with pytest.raises(ValueError, match="temperature nulle ou negative"):
        fraction_emise_sous(1e-6, 0.0)


def test_une_longueur_d_onde_nulle_n_a_pas_de_fraction_emise() -> None:
    with pytest.raises(ValueError, match="longueur d'onde nulle"):
        fraction_emise_sous(0.0, 300.0)


def test_une_emissivite_nulle_est_refusee_entre_deux_plans() -> None:
    with pytest.raises(ValueError, match="emissivite hors de"):
        flux_entre_deux_plans_gris(0.0, 0.8, 500.0, 300.0, aire=1.0)


def test_une_aire_nulle_est_refusee_entre_deux_plans() -> None:
    with pytest.raises(ValueError, match="aire nulle ou negative"):
        flux_entre_deux_plans_gris(0.8, 0.8, 500.0, 300.0, aire=0.0)


def test_un_solenoide_de_geometrie_nulle_est_refuse() -> None:
    with pytest.raises(ValueError, match="geometrie invalide"):
        inductance_propre_d_un_solenoide(100, 0.0, 0.1)


def test_un_solenoide_de_permeabilite_nulle_est_refuse() -> None:
    with pytest.raises(ValueError, match="permeabilite relative"):
        inductance_propre_d_un_solenoide(100, 1e-4, 0.1, relative=0.0)


def test_un_volume_negatif_est_refuse_dans_l_hysteresis() -> None:
    with pytest.raises(ValueError, match="negatif"):
        pertes_par_hysteresis(200.0, 50.0, -1e-3)
