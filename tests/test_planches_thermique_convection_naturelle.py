"""Les planches de thermique : convection naturelle et coefficient global (jeu 10).

Source : `ThermiqueFondements/SLIDES/Slides_10_Convection_naturelle.pdf`,
29 planches lues sur les pages rendues. Elles reprennent le chapitre 4
de la premiere partie du tome 2 et le chapitre 1 de la seconde.

Les oracles : les correlations des planches 3 a 8, la synthese de la
planche 9, le graphique de la tablette (planche 22, lu au pixel : 13,9 W
debout, 7,9 W a plat sur un support adiabatique) et le tableau du boiler
(planche 27), recalcule colonne par colonne.
"""

from __future__ import annotations

import math

import pytest

from polytech.fluides.domain.proprietes import (
    coefficient_de_dilatation_du_gaz_parfait,
)
from polytech.maths.domain.point_fixe.approximations_successives import (
    approximations_successives,
)
from polytech.maths.domain.point_fixe.contraction import iterer
from polytech.thermique.domain.coefficient_global import (
    coefficient_global_plan,
    temperature_de_paroi,
)
from polytech.thermique.domain.convection.forcee import coefficient_d_echange
from polytech.thermique.domain.convection.naturelle import (
    CONDUCTION_PURE,
    grashof,
    nusselt_du_cylindre_horizontal,
    nusselt_lame_de_fluide,
)
from polytech.thermique.domain.convection.naturelle_externe import (
    GRASHOF_CRITIQUE,
    hauteur_critique,
    nusselt_de_la_paroi_verticale,
    nusselt_de_la_plaque_horizontale,
    nusselt_du_cylindre,
)
from polytech.thermique.domain.convection.similitude import masse_volumique_dilatee
from polytech.thermique.domain.convection.synthese_naturelle import (
    CorrelationNaturelle,
)
from polytech.thermique.domain.ordres_de_grandeur import ORDRES_DE_GRANDEUR, Phenomene
from polytech.thermique.domain.paroi_de_separation import chutes_partielles
from polytech.thermique.domain.rayonnement_equivalent import coefficient_entre_surfaces
from polytech.thermique.domain.tables.air_sec import air_sec
from polytech.thermique.domain.tables.eau_saturee import eau_saturee


def test_le_coefficient_de_dilatation_volumique() -> None:
    """Planche 2 : `rho = rho_0 (1 - beta theta)` ; gaz parfait, 3,33e-3 a 300 K."""
    assert coefficient_de_dilatation_du_gaz_parfait(300.0) == pytest.approx(
        3.33e-3, abs=5e-6
    )
    assert masse_volumique_dilatee(1000.0, 2e-4, 10.0) == pytest.approx(998.0)


def test_la_paroi_verticale() -> None:
    """Planche 3 : laminaire jusqu'a `Gr = 1e9`, `0,56 Ra^1/4` puis `0,12 Ra^1/3`."""
    assert pytest.approx(1e9) == GRASHOF_CRITIQUE
    critique = hauteur_critique(1 / 300.0, 30.0, 1.6e-5)
    assert grashof(1 / 300.0, 30.0, critique, 1.6e-5) == pytest.approx(1e9)
    assert nusselt_de_la_paroi_verticale(1e7) == pytest.approx(0.56 * 1e7**0.25)
    assert nusselt_de_la_paroi_verticale(1e10) == pytest.approx(0.12 * 1e10 ** (1 / 3))


def test_le_cylindre_horizontal() -> None:
    """Planche 4 : `0,47 (Gr Pr)^1/4` de 1e3 a 1e9, `0,10 (Gr Pr)^1/3` au-dela."""
    assert nusselt_du_cylindre(1e6, 0.7) == pytest.approx(0.47 * 7e5**0.25)
    assert nusselt_du_cylindre(1e10, 0.7) == pytest.approx(0.10 * 7e9 ** (1 / 3))
    with pytest.raises(ValueError, match="hors du domaine des relations"):
        nusselt_du_cylindre(500.0, 0.7)


def test_la_surface_horizontale_ne_depend_pas_de_sa_longueur() -> None:
    """Planche 5 : `Nu = 0,137 (Gr_L Pr)^1/3`, donc `K` independant de `L`."""
    echanges = []
    for longueur in (0.2, 0.4):
        rayleigh = grashof(1 / 300.0, 20.0, longueur, 1.6e-5) * 0.7
        valeur = nusselt_de_la_plaque_horizontale(rayleigh)
        assert valeur == pytest.approx(0.137 * rayleigh ** (1 / 3))
        echanges.append(coefficient_d_echange(valeur, 0.026, longueur))
    assert echanges[0] == pytest.approx(echanges[1])


def test_la_lame_de_fluide() -> None:
    """Planches 7 et 8 : conduction pure sous 1400, puis `C (Gr_b Pr)^n`.

    « n ~ 0,3 ... 0,2 » : la bibliotheque porte 0,29 puis 0,20, ceux du
    tome 2.
    """
    assert pytest.approx(1400.0) == CONDUCTION_PURE
    assert nusselt_lame_de_fluide(1000.0) == 1.0
    assert nusselt_lame_de_fluide(1e5) == pytest.approx(0.11 * 1e5**0.29)
    assert nusselt_lame_de_fluide(1e7) == pytest.approx(0.40 * 1e7**0.20)
    assert nusselt_lame_de_fluide(1e5) > 1


VERTICALE = CorrelationNaturelle(coefficient=0.56, exposant=0.25)
HORIZONTALE = CorrelationNaturelle(coefficient=0.137, exposant=1 / 3)


def test_la_synthese_de_la_convection_naturelle() -> None:
    """Planche 9 : `K ~ theta_0^n`, `phi ~ Dt^(1 + n)`, `K ~ l^(3n - 1)`."""
    assert VERTICALE.nusselt(1e7) == pytest.approx(nusselt_de_la_paroi_verticale(1e7))
    assert VERTICALE.effet_de_l_ecart(2.0) == pytest.approx(2.0**0.25)
    assert VERTICALE.effet_sur_le_flux(2.0) == pytest.approx(2.0**1.25)
    assert VERTICALE.effet_sur_le_flux(2.0) > 2.0
    assert VERTICALE.effet_de_la_dimension(2.0) == pytest.approx(2.0**-0.25)
    assert HORIZONTALE.effet_de_la_dimension(2.0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="exposant hors de"):
        CorrelationNaturelle(coefficient=0.5, exposant=0.5)
    with pytest.raises(ValueError, match="rapport nul ou negatif"):
        VERTICALE.effet_de_l_ecart(0.0)


def test_la_synthese_se_verifie_sur_la_correlation() -> None:
    """Doubler `l` multiplie `Gr` par huit, donc `Nu` par `8^n` et `K` par `8^n / 2`."""
    rapport = VERTICALE.nusselt(8e7) / VERTICALE.nusselt(1e7) / 2
    assert rapport == pytest.approx(VERTICALE.effet_de_la_dimension(2.0))
    assert VERTICALE.nusselt(2e7) / VERTICALE.nusselt(1e7) == pytest.approx(
        VERTICALE.effet_de_l_ecart(2.0)
    )


def test_les_ordres_de_grandeur_de_la_planche_17() -> None:
    """Eau en convection forcee : 4e3 a 8e3, comme le syllabus.

    Pour la convection naturelle la planche donne des plages plus
    larges que le tome 2 : 1 a 10 dans l'air (2 a 5), 1e2 a 1e3 dans
    l'eau (25 a 500). Elles se recouvrent.
    """
    assert ORDRES_DE_GRANDEUR[Phenomene.EAU_TURBULENTE_EN_TUBE] == (4e3, 8e3)
    air = ORDRES_DE_GRANDEUR[Phenomene.CONVECTION_NATURELLE_DANS_L_AIR]
    assert 1.0 <= air[0] < air[1] <= 10.0
    eau = ORDRES_DE_GRANDEUR[Phenomene.CONVECTION_NATURELLE_DANS_L_EAU]
    assert eau[0] < 1e2 < eau[1] < 1e3


def test_la_temperature_de_paroi_se_deduit_du_coefficient_global() -> None:
    """Planche 18 : `t_p1 = t_1 - (K_gl / K_1)(t_1 - t_2)`."""
    premier, second, chaud, froid = 20.0, 500.0, 80.0, 20.0
    global_ = coefficient_global_plan(premier, 0.002, 45.0, second)
    paroi = chaud - global_ / premier * (chaud - froid)
    chutes = chutes_partielles(chaud - froid, (1 / premier, 0.002 / 45.0, 1 / second))
    assert paroi == pytest.approx(chaud - chutes[0])
    assert paroi == pytest.approx(
        temperature_de_paroi(chaud, global_ * (chaud - froid), premier)
    )


def test_les_approximations_successives() -> None:
    """Planche 18 : reprendre le calcul jusqu'a `|K_gl - K_gl*| < epsilon`."""
    valeur, tours = approximations_successives(math.cos, 1.0, tolerance=1e-12)
    assert valeur == pytest.approx(0.7390851332151607, abs=1e-11)
    assert tours > 10
    assert iterer(math.cos, 1.0, tours)[-1] == pytest.approx(valeur)
    assert approximations_successives(lambda x: x / 2 + 1, 0.0, tolerance=0.3) == (
        1.75,
        3,
    )
    with pytest.raises(ValueError, match="pas de convergence"):
        approximations_successives(
            lambda x: 2 * x + 1, 1.0, tolerance=1e-12, iterations=20
        )


SURFACE, PAROI, PIECE = 0.24 * 0.17, 41.0, 25.0


def _echange_de_la_tablette(nusselt: float, longueur: float) -> float:
    film = air_sec(273.15 + (PAROI + PIECE) / 2)
    convection = coefficient_d_echange(nusselt, film.conductivite, longueur)
    rayonnement = coefficient_entre_surfaces(0.96, 273.15 + PAROI, 273.15 + PIECE)
    return convection + rayonnement


def _rayleigh_de_la_tablette(longueur: float) -> float:
    moyenne = 273.15 + (PAROI + PIECE) / 2
    film = air_sec(moyenne)
    nombre = grashof(
        coefficient_de_dilatation_du_gaz_parfait(moyenne),
        PAROI - PIECE,
        longueur,
        film.viscosite_cinematique(),
    )
    return nombre * film.prandtl


def test_la_tablette_debout() -> None:
    """Planches 20 a 22 : 41 degres au plus, dans une piece a 25 degres.

    La simulation de la planche 22 donne 13,9 W debout. Les relations du
    cours -- paroi verticale de 0,24 m, deux faces, rayonnement d'une
    surface d'emissivite 0,96 -- donnent 13,6 W.
    """
    rayleigh = _rayleigh_de_la_tablette(0.24)
    echange = _echange_de_la_tablette(nusselt_de_la_paroi_verticale(rayleigh), 0.24)
    flux = echange * 2 * SURFACE * (PAROI - PIECE)
    assert flux == pytest.approx(13.9, rel=0.05)
    assert flux == pytest.approx(13.6, abs=0.15)


def test_la_tablette_a_plat_sur_un_support_adiabatique() -> None:
    """Planche 22 : 7,9 W lus sur le graphique ; les relations du cours, 6,7 W.

    Une seule face echange. L'ecart, 15 %, est celui d'une correlation
    de plaque infinie appliquee a une plaque dont les bords aspirent.
    """
    rayleigh = _rayleigh_de_la_tablette(0.24)
    echange = _echange_de_la_tablette(nusselt_de_la_plaque_horizontale(rayleigh), 0.24)
    flux = echange * SURFACE * (PAROI - PIECE)
    assert flux == pytest.approx(7.9, rel=0.2)
    assert flux == pytest.approx(6.7, abs=0.15)


PUISSANCE, DIAMETRE, EAU = 1800.0, 0.05, 50.0
AIRE = 0.04948
COLONNES = (
    (800.0, 95.47, (3.78e-4, 0.671, 2.36, 974.60, 5.85e-4), 948.30, 88.36),
    (948.30, 88.36, (4.04e-4, 0.668, 2.54, 977.52, 5.87e-4), 893.66, 90.71),
    (893.66, 90.71, (4.015e-4, 0.668, 2.54, 977.52, 5.87e-4), 910.06, 89.97),
)
"""Planche 27 : `K_c*`, `t_p*`, (`mu`, `lambda`, `Pr`, `rho`, `beta`), `K_c`, `t_p`.

La troisieme viscosite est imprimee 4,01e-4 ; 4,015e-4 rend son `Gr Pr`.
"""


def _resistance(echange: float) -> float:
    return temperature_de_paroi(EAU, -PUISSANCE / AIRE, echange)


def test_le_boiler_est_calcule_pour_cinquante_millimetres() -> None:
    """Planche 27 : `S = 0,04948 m^2`, soit `pi x 0,05 x 0,315`.

    L'enonce des planches 25 et 27 donne `d = 45 mm` ; le tableau porte
    `d = 0,05` et tous ses nombres en decoulent. Avec 45 mm, 0,0445 m^2.
    """
    assert math.pi * DIAMETRE * 0.315 == pytest.approx(AIRE, abs=5e-6)
    assert math.pi * 0.045 * 0.315 == pytest.approx(0.0445, abs=5e-5)


@pytest.mark.parametrize(("depart", "paroi", "eau", "echange", "suivante"), COLONNES)
def test_le_tableau_du_boiler_colonne_par_colonne(
    depart: float,
    paroi: float,
    eau: tuple[float, float, float, float, float],
    echange: float,
    suivante: float,
) -> None:
    """Chaque colonne : `t_p*` de `K_c*`, `Gr Pr`, `Nu = 0,47 Ra^1/4`, `K_c`, `t_p`."""
    viscosite, conductivite, prandtl, masse_volumique, dilatation = eau
    assert _resistance(depart) == pytest.approx(paroi, abs=0.01)
    nombre = grashof(dilatation, paroi - EAU, DIAMETRE, viscosite / masse_volumique)
    nusselt = nusselt_du_cylindre_horizontal(nombre * prandtl)
    calcule = coefficient_d_echange(nusselt, conductivite, DIAMETRE)
    assert calcule == pytest.approx(echange, rel=2e-3)
    assert _resistance(calcule) == pytest.approx(suivante, abs=0.1)


def test_le_boiler_converge_vers_quatre_vingt_dix_degres() -> None:
    """La meme iteration, menee a son terme avec la table de l'eau du depot."""

    def pas(echange: float) -> float:
        paroi = _resistance(echange)
        film = eau_saturee((paroi + EAU) / 2)
        nombre = grashof(5.87e-4, paroi - EAU, DIAMETRE, film.viscosite_cinematique)
        nusselt = nusselt_du_cylindre_horizontal(nombre * film.prandtl)
        return coefficient_d_echange(nusselt, film.conductivite, DIAMETRE)

    echange, _ = approximations_successives(pas, 800.0, tolerance=1e-6)
    assert _resistance(echange) == pytest.approx(90.0, abs=0.5)
    assert echange == pytest.approx(910.0, rel=0.02)
