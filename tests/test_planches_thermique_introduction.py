"""Les planches de thermique : l'introduction.

Source : `ThermiqueFondements/SLIDES/Slides_01_Introduction.pdf`, les 31
planches, lues une a une.

Les oracles : les deux modes fondamentaux de la planche 19, l'alternateur
de la planche 12, l'echelle de Newton de la planche 27, les quatre points
de la planche 28, les schemas des planches 29 et 31.
"""

from __future__ import annotations

import math

import pytest

from polytech.thermique.domain.coefficient_global import (
    coefficient_de_rayonnement_exact,
)
from polytech.thermique.domain.echelle_de_newton import (
    celsius_depuis_degres_newton,
    degres_newton_depuis_celsius,
)
from polytech.thermique.domain.modes import (
    ModeDeConvection,
    coefficient_superpose,
    flux_de_newton,
    mode_de_convection,
    puissance_metabolique_au_repos,
    resistance_depuis_l_echange,
)
from polytech.thermique.domain.modes_fondamentaux import (
    ModeFondamental,
    Transmission,
    chaleur_de_l_activite,
    couche_au_repos,
    puissance_a_evacuer,
    temperature_d_equilibre,
)
from polytech.thermique.domain.refroidissement import (
    delai_de_refroidissement,
    taux_de_refroidissement,
    temperature_du_corps,
)
from polytech.thermique.domain.reseaux import en_parallele, en_serie


def test_la_thermodynamique_predit_l_equilibre() -> None:
    """Planche 5 : le premier principe donne `t_m`, pas le temps d'y arriver."""
    cuivre, eau = 0.5 * 385.0, 2.0 * 4180.0
    equilibre = temperature_d_equilibre((cuivre, eau), (200.0, 20.0))
    assert 20.0 < equilibre < 200.0
    cede = cuivre * (200.0 - equilibre)
    recu = eau * (equilibre - 20.0)
    assert cede == pytest.approx(recu)
    assert temperature_d_equilibre((cuivre, eau), (50.0, 50.0)) == pytest.approx(50.0)
    with pytest.raises(ValueError, match="capacite nulle ou negative"):
        temperature_d_equilibre((0.0, eau), (200.0, 20.0))


def test_la_resistance_thermique() -> None:
    """Planches 7 et 25 : `Phi = (t_1 - t_2) / R_th`, `R_th = 1 / (K S)`."""
    resistance = resistance_depuis_l_echange(25.0, 2.0)
    assert resistance == pytest.approx(0.02)
    assert flux_de_newton(25.0, 2.0, 30.0) == pytest.approx(30.0 / resistance)
    assert flux_de_newton(25.0, 2.0, 0.0) == 0.0


def test_un_alternateur_de_1000_mw_dissipe_20_mw() -> None:
    """Planche 12 : « eta = 0,98 => 20 MW a evacuer »."""
    assert puissance_a_evacuer(1000e6, 0.98) == pytest.approx(20e6)
    with pytest.raises(ValueError, match="rendement hors de"):
        puissance_a_evacuer(1000e6, 1.2)


def test_la_dissipation_du_corps_humain() -> None:
    """Planche 16 : 100 W au repos ; a l'effort, un rendement de 0,2 a 0,25."""
    assert puissance_metabolique_au_repos() == 100.0
    assert chaleur_de_l_activite(100.0, 0.25) == pytest.approx(300.0)
    assert chaleur_de_l_activite(100.0, 0.2) == pytest.approx(400.0)
    with pytest.raises(ValueError, match="rendement hors de"):
        chaleur_de_l_activite(100.0, 0.0)


def test_il_y_a_deux_modes_fondamentaux() -> None:
    """Planche 19 : conductibilite et rayonnement ; la convection est du premier."""
    assert len(ModeFondamental) == 2
    assert Transmission.CONDUCTIBILITE_PURE.mode is ModeFondamental.CONDUCTIBILITE
    assert Transmission.CONVECTION.mode is ModeFondamental.CONDUCTIBILITE
    assert Transmission.RAYONNEMENT.mode is ModeFondamental.RAYONNEMENT


def test_seul_le_rayonnement_traverse_le_vide() -> None:
    """Planches 19 et 20 : « necessite un milieu materiel », « OK dans le vide »."""
    assert ModeFondamental.CONDUCTIBILITE.exige_un_milieu
    assert not ModeFondamental.RAYONNEMENT.exige_un_milieu
    assert Transmission.CONVECTION.milieu_en_mouvement
    assert not Transmission.CONDUCTIBILITE_PURE.milieu_en_mouvement


def test_les_modes_de_convection_de_la_planche_22() -> None:
    """Un ventilateur a 10 m/s domine ; sans lui, ce sont les forces d'Archimede."""
    assert mode_de_convection(force=80.0, naturel=5.0) is ModeDeConvection.FORCEE
    assert mode_de_convection(force=0.3, naturel=5.0) is ModeDeConvection.NATURELLE
    assert mode_de_convection(force=8.0, naturel=5.0) is ModeDeConvection.MIXTE


def test_une_couche_chauffee_par_le_dessus_reste_au_repos() -> None:
    """Planche 23 : `t_1 > t_2` est stable, conduction pure ; l'inverse convecte."""
    assert couche_au_repos(dessus=60.0, dessous=20.0)
    assert not couche_au_repos(dessus=20.0, dessous=60.0)
    assert couche_au_repos(dessus=20.0, dessous=20.0)


def test_l_echelle_de_newton() -> None:
    """Planche 27 : 0 et 12 degres N ; 34 degres N, 104 ; l'etain, 72 degres N, 222."""
    assert degres_newton_depuis_celsius(0.0) == 0.0
    assert degres_newton_depuis_celsius(37.0) == pytest.approx(12.0)
    assert celsius_depuis_degres_newton(34.0) == pytest.approx(104.0, abs=1.0)
    etain = celsius_depuis_degres_newton(72.0)
    assert etain == pytest.approx(222.0)
    assert etain < 232.0


POINTS = ((1.82, 83.0), (2.36, 59.0), (3.52, 35.0), (4.52, 21.0))
"""Planche 28 : les quatre carres noirs de la courbe `t - t_a`, mesures au pixel."""


def test_la_loi_de_refroidissement_aligne_les_logarithmes() -> None:
    """Planche 28 : `d(t - t_a)/dtau = -alpha (t - t_a)`, depuis 200 degres."""
    taux = [
        taux_de_refroidissement(200.0, ecart, 0.0, instant) for instant, ecart in POINTS
    ]
    assert max(taux) - min(taux) < 0.04
    moyen = sum(taux) / len(taux)
    assert moyen == pytest.approx(0.5, abs=0.02)
    for instant, ecart in POINTS:
        assert temperature_du_corps(200.0, 0.0, moyen, instant) == pytest.approx(
            ecart, abs=3.0
        )


def test_le_second_graphe_est_en_logarithmes_decimaux() -> None:
    """Planche 28 : l'axe est note « ln(t - t_a) », mais il part de 2,3.

    `log10(200) = 2,30` ; `ln(200) = 5,30`. Les quatre points, lus a 1,9,
    1,77, 1,55 et 1,34, sont les logarithmes decimaux des ecarts.
    """
    assert math.log10(200.0) == pytest.approx(2.3, abs=0.005)
    assert math.log(200.0) == pytest.approx(5.3, abs=0.005)
    lus = (1.92, 1.77, 1.55, 1.34)
    for (_, ecart), lu in zip(POINTS, lus, strict=True):
        assert math.log10(ecart) == pytest.approx(lu, abs=0.02)
        assert math.log(ecart) > lu + 1.7


def test_l_heure_du_deces_par_la_loi_de_newton() -> None:
    """Planche 30 : deux temperatures du corps a deux instants donnent l'heure."""
    taux = taux_de_refroidissement(30.0, 28.0, 20.0, 1.0)
    delai = delai_de_refroidissement(37.0, 30.0, 20.0, taux)
    assert temperature_du_corps(37.0, 20.0, taux, delai) == pytest.approx(30.0)
    assert temperature_du_corps(37.0, 20.0, taux, delai + 1.0) == pytest.approx(28.0)


def test_l_experience_de_newton_met_deux_modes_en_parallele() -> None:
    """Planche 29 : `phi = (K_conv + K_ray)(t_p - t_a) = (t_p - t_a) / R_th`."""
    convection, rayonnement = 25.0, 40.0
    ensemble = coefficient_superpose((convection, rayonnement))
    assert ensemble == pytest.approx(65.0)
    resistance = en_parallele(1 / convection, 1 / rayonnement)
    assert 1 / resistance == pytest.approx(ensemble)


def test_la_loi_de_newton_ne_vaut_pas_pour_le_rayonnement() -> None:
    """Planche 30 : le coefficient de rayonnement depend des temperatures."""
    tiede = coefficient_de_rayonnement_exact(0.8, 350.0, 293.0)
    rouge = coefficient_de_rayonnement_exact(0.8, 1100.0, 293.0)
    assert rouge > 10 * tiede


def test_l_aerotherme_met_trois_resistances_en_serie() -> None:
    """Planche 31 : l'eau, la paroi et l'air en serie ; `Phi = (t_e - t_a) / R_th`."""
    eau = resistance_depuis_l_echange(3000.0, 0.05)
    paroi = 2.0e-4
    air = resistance_depuis_l_echange(60.0, 0.06)
    totale = en_serie(eau, paroi, air)
    flux = (80.0 - 20.0) / totale
    interieure = 80.0 - flux * eau
    exterieure = interieure - flux * paroi
    assert flux_de_newton(60.0, 0.06, exterieure - 20.0) == pytest.approx(flux)
    assert air > 10 * eau
