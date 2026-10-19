"""Activite d'integration 2 : les hypotheses du corrige, chiffrees.

Chaque simplification que le corrige fait est mesuree au lieu d'etre
crue. Les donnees viennent du cours :

- l'huile : `MecaniqueFluides/SLIDES/Slides_07_Lubrification_Ecoulement_turbulent.pdf`,
  planche 14, « Huile SAE50 a 50 degres », rho = 900 kg/m3, mu = 0,1 N s/m2 ;
- l'air : `Slides_01_Introduction_2025_2026.pdf`, mu = 1,83 . 10^-5 Pa s a
  vingt degres et une atmosphere.

Oracles calcules a la main ici : 1,83e-5 / 0,1 = 1,83e-4 ;
rho_air (101325 Pa, 293,15 K, 29 g/mol) = 1,2056 kg/m3, et 1,2056 / 900 =
1,34e-3 ; `Re = rho^2 g h^3 / (3 mu^2)` vaut 0,2649 pour h = 1 mm.
"""

from __future__ import annotations

import math

import pytest

from polytech.fluides.application import activite_2_hypotheses as hyp


def test_l_air_a_cinq_mille_fois_moins_de_viscosite_que_l_huile() -> None:
    assert hyp.rapport_des_viscosites() == pytest.approx(1.83e-4)


def test_la_poussee_de_l_air_allege_l_huile_d_un_millieme() -> None:
    assert hyp.rapport_des_masses_volumiques() == pytest.approx(1.2056 / 900, rel=1e-3)


def test_le_reynolds_du_film_d_un_millimetre() -> None:
    attendu = 900**2 * 9.81 * 1e-9 / (3 * 0.1**2)
    assert hyp.reynolds_du_film(1e-3) == pytest.approx(attendu)
    assert hyp.reynolds_du_film(1e-3) == pytest.approx(0.2649, rel=1e-3)


def test_le_reynolds_croit_comme_le_cube_de_l_epaisseur() -> None:
    assert hyp.reynolds_du_film(2e-3) / hyp.reynolds_du_film(1e-3) == pytest.approx(8)


def test_l_epaisseur_ou_l_inertie_egale_la_viscosite() -> None:
    epaisseur = hyp.epaisseur_a_reynolds_unite()
    assert hyp.reynolds_du_film(epaisseur) == pytest.approx(1.0)
    assert epaisseur == pytest.approx((3 * 0.1**2 / (900**2 * 9.81)) ** (1 / 3))


def test_une_epaisseur_nulle_leve() -> None:
    with pytest.raises(ValueError, match="epaisseur de film non strictement positive"):
        hyp.reynolds_du_film(0.0)


def test_la_vitesse_de_la_surface_du_reservoir_retire_r_puissance_4() -> None:
    """Bernoulli complet : `V_o^2 (1 - r^4) = 2 g z`, donc `z_max (1 - r^4)`."""
    assert hyp.facteur_de_vitesse_de_surface(0.1) == pytest.approx(1 - 1e-4)
    assert hyp.facteur_de_vitesse_de_surface(0.5) == pytest.approx(15 / 16)


@pytest.mark.parametrize("rapport", [0.0, 1.0, -0.2, 1.5])
def test_un_orifice_hors_du_reservoir_leve(rapport: float) -> None:
    with pytest.raises(ValueError, match="rapport des diametres hors de"):
        hyp.facteur_de_vitesse_de_surface(rapport)


def test_une_perte_singuliere_eleve_la_hauteur_d_autant() -> None:
    """`g z = (1 + K) V_o^2 / 2` : la hauteur est multipliee par `1 + K`."""
    assert hyp.facteur_de_perte(0.05) == pytest.approx(1.05)
    assert hyp.facteur_de_perte(0.0) == 1.0


def test_une_perte_negative_leve() -> None:
    with pytest.raises(ValueError, match="coefficient de perte negatif"):
        hyp.facteur_de_perte(-0.1)


def test_les_elasticites_de_la_puissance_du_bateau() -> None:
    """`P = rho g Q (dz + h_L + 8 Q^2 / (pi^2 g D^4)) / eta`, derivee en log."""
    part = (50.93**2 / (2 * 9.81)) / 139.2
    elasticites = hyp.elasticites_du_bateau()
    assert elasticites["diametre"] == pytest.approx(-4 * part, rel=1e-3)
    assert elasticites["debit"] == pytest.approx(1 + 2 * part, rel=1e-3)
    assert elasticites["rendement"] == pytest.approx(-1)
    assert elasticites["masse_volumique"] == pytest.approx(1)


def test_un_millimetre_sur_la_buse_par_le_calcul_complet() -> None:
    """Le recoupement de l'elasticite : on refait le bateau a 49 et 51 mm."""
    ecarts = hyp.ecarts_pour_un_millimetre_de_buse()
    assert ecarts[0] > 0 > ecarts[1]
    moyenne = (ecarts[0] - ecarts[1]) / 2
    assert moyenne == pytest.approx(
        -hyp.elasticites_du_bateau()["diametre"] * 0.02, rel=0.02
    )
    assert math.isclose(abs(ecarts[0]), 0.081, rel_tol=0.05)
