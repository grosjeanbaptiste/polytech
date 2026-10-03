"""Le TP 4 de thermique : le nombre de rangs de tubes ailettes.

L'officiel dimensionne la batterie TCPL16-40-4-50 et trouve 7,514 rangs,
donc huit. Ce fichier refait chaque etape et la compare a la sienne,
puis verifie le dimensionnement par un second chemin : avec la
conductance trouvee, la methode NUT-efficacite a contre-courant doit
redonner exactement le flux vise.
"""

from __future__ import annotations

import pytest

from polytech.thermique.application.dimensionnement_tp_04 import dimensionner
from polytech.thermique.application.enonce_tp_04 import bilan, flux_par_nut
from polytech.thermique.application.proprietes_tp_04 import BATTERIES


def test_le_bilan_donne_la_sortie_de_l_eau_et_le_flux() -> None:
    sortie, flux, _ = bilan()
    assert sortie == pytest.approx(17.067, abs=1e-3)
    assert flux == pytest.approx(12084, abs=1)


def test_la_conductance_est_le_flux_sur_la_moyenne_logarithmique() -> None:
    """775,51 W/K ; l'officiel ecrit 775,61, a un dix-millieme pres."""
    _, flux, conductance = bilan()
    assert conductance == pytest.approx(flux / 15.5826, rel=1e-4)
    assert conductance == pytest.approx(775.61, rel=2e-4)


@pytest.mark.parametrize(
    ("grandeur", "attendu", "tolerance"),
    [
        ("nusselt_interieur", 108.033, 0.05),
        ("convection_interieure", 4692.3, 2.0),
        ("reynolds_air", 1539.41, 0.5),
        ("nusselt_air", 11.48, 0.01),
        ("convection_air", 67.212, 0.05),
        ("parametre", 40.004, 0.01),
        ("argument", 0.5828, 1e-4),
        ("efficacite_droite", 0.9003, 1e-4),
        ("efficacite", 0.835, 5e-4),
        ("coefficient_global", 671.17, 0.2),
        ("surface_interieure", 1.1556, 5e-4),
        ("longueur", 26.3, 0.05),
        ("rangs_exacts", 7.506, 5e-3),
    ],
)
def test_chaque_etape_rejoint_l_officiel(
    grandeur: str, attendu: float, tolerance: float
) -> None:
    resultat = dimensionner("TCPL16-40-4-50")
    assert getattr(resultat, grandeur) == pytest.approx(attendu, abs=tolerance)


def test_les_7_514_rangs_de_l_officiel_viennent_de_sa_longueur_arrondie() -> None:
    """26,3 m / 3,5 m par rang = 7,514 ; la longueur exacte donne 7,506."""
    assert pytest.approx(7.514, abs=5e-4) == 26.3 / (7 * 0.5)


def test_il_faut_huit_rangs() -> None:
    assert dimensionner("TCPL16-40-4-50").rangs == 8


def test_la_methode_nut_redonne_le_flux_vise() -> None:
    _, flux, conductance = bilan()
    assert flux_par_nut(conductance) == pytest.approx(flux, rel=1e-9)


def test_les_quatre_batteries_se_dimensionnent() -> None:
    assert all(dimensionner(nom).rangs >= 1 for nom in BATTERIES)
