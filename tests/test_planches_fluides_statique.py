"""Planches de mecanique des fluides, chapitre II : statique des fluides.

Source : `MecaniqueFluides/SLIDES/Slides_02_Statique_Cinematique_2025_2026.pdf`,
planches 3 a 19, lues sur les pages rendues.

Les oracles du cours : les ordres de grandeur des planches 8 et 9 (dix
metres d'eau pour un bar, la fosse des Mariannes), et l'exercice du
manometre multi-fluides, dont la reponse imprimee est 130 kPa. La porte
cylindrique de la planche 19 est donnee sans reponse : elle est resolue
par les projections, puis recoupee par l'integrale de la pression le long
de l'arc.
"""

from __future__ import annotations

import math

import pytest

from polytech.fluides.domain.grandeurs import Hauteur, MasseVolumique, Pression
from polytech.fluides.domain.statique.manometre import Manometre, Sens, Troncon
from polytech.fluides.domain.statique.pressions import (
    ATMOSPHERE,
    BAR,
    en_atmospheres,
    en_bars,
    force_de_la_presse,
    hauteur_barometrique,
    pression_absolue,
    pression_relative,
    vide,
)
from polytech.fluides.domain.statique.stevin import variation_de_pression
from polytech.fluides.domain.statique.surface_courbe import PorteCylindrique

EAU = MasseVolumique(kg_par_m3=1000.0)
HUILE = MasseVolumique(kg_par_m3=850.0)
MERCURE = MasseVolumique(kg_par_m3=13_600.0)
G_ARRONDI = 10.0
"""Les planches 8 et 9 prennent `g = 10 m/s^2`."""


def test_les_unites_de_pression_de_la_planche_4() -> None:
    """`1 bar = 10^5 Pa`, `1 atm = 101 325 Pa`."""
    assert BAR == 1e5
    assert ATMOSPHERE == 101_325.0


def test_les_conversions_en_bars_et_en_atmospheres() -> None:
    """Planche 4 : une atmosphere vaut 1,01325 bar ; un bar, 0,98692 atm."""
    assert en_bars(ATMOSPHERE) == pytest.approx(1.01325, rel=1e-12)
    assert en_atmospheres(BAR) == pytest.approx(0.986923, abs=5e-7)
    assert en_bars(0.0) == 0.0
    assert en_atmospheres(en_bars(ATMOSPHERE) * BAR) == pytest.approx(1.0, rel=1e-12)


def test_pression_relative_et_vide_se_mesurent_depuis_l_atmosphere() -> None:
    """Planche 4 : `p_gauge = p_abs - p_atm`, `p_vac = p_atm - p_abs`."""
    assert pression_relative(201_325.0, ATMOSPHERE) == pytest.approx(1e5)
    assert vide(60_000.0, ATMOSPHERE) == pytest.approx(41_325.0)
    assert pression_absolue(1e5, ATMOSPHERE) == pytest.approx(201_325.0)
    assert pression_relative(60_000.0, ATMOSPHERE) == -vide(60_000.0, ATMOSPHERE)


def test_une_pression_absolue_negative_est_refusee() -> None:
    with pytest.raises(ValueError, match="pression absolue negative"):
        pression_relative(-1.0, ATMOSPHERE)
    with pytest.raises(ValueError, match="pression absolue negative"):
        pression_absolue(-2e5, ATMOSPHERE)
    with pytest.raises(ValueError, match="pas un vide"):
        vide(2e5, ATMOSPHERE)


def test_dix_metres_d_eau_pesent_un_bar() -> None:
    """Planche 8 : `z_1 = 11 m`, `z_2 = 1 m`, `P_2 relatif = rho g h = 1 bar`."""
    relatif = variation_de_pression(EAU, Hauteur(metres=11.0 - 1.0), G_ARRONDI)
    assert relatif.pascals == pytest.approx(BAR)
    absolue = pression_absolue(relatif.pascals, ATMOSPHERE)
    assert absolue == pytest.approx(201_325.0)


def test_la_fosse_des_mariannes() -> None:
    """Planche 9 : 110 MPa a 11 km, 440 MN sur 4 m^2, 11 kN par cm^2.

    La planche ecrit « 1100 atm » : 110 MPa font 1 100 bar, et 1 086 atm.
    """
    pression = variation_de_pression(EAU, Hauteur(metres=11_000.0), G_ARRONDI)
    assert pression.pascals == pytest.approx(110e6)
    assert pression.pascals * 4 == pytest.approx(440e6)
    assert pression.pascals * 1e-4 == pytest.approx(11e3)
    assert en_bars(pression.pascals) == pytest.approx(1100.0)
    assert round(en_atmospheres(pression.pascals)) == 1086


def test_la_presse_hydraulique_multiplie_la_force_par_le_rapport_des_aires() -> None:
    """Planche 11 : `P_1 = P_2`, donc `F_2 = F_1 A_2 / A_1`."""
    assert force_de_la_presse(100.0, 0.001, 0.1) == pytest.approx(10_000.0)
    # Soulever une voiture de 1,2 tonne « a une main » : 118 N suffisent.
    poids = 1200.0 * 9.81
    assert force_de_la_presse(poids, 0.1, 0.001) == pytest.approx(117.7, abs=0.1)


def test_un_piston_sans_aire_est_refuse() -> None:
    with pytest.raises(ValueError, match="aire nulle ou negative"):
        force_de_la_presse(100.0, 0.0, 0.1)
    with pytest.raises(ValueError, match="aire nulle ou negative"):
        force_de_la_presse(100.0, 0.1, 0.0)


def test_le_barometre_de_torricelli_monte_a_760_millimetres() -> None:
    """Planche 12 : `P_atm = rho g h`, avec du mercure."""
    hauteur = hauteur_barometrique(ATMOSPHERE, MERCURE)
    assert hauteur.metres == pytest.approx(0.760, abs=1e-3)
    retour = variation_de_pression(MERCURE, hauteur)
    assert retour.pascals == pytest.approx(ATMOSPHERE)
    # Avec de l'eau, il faudrait un tube de dix metres : c'est la planche 8.
    avec_de_l_eau = hauteur_barometrique(ATMOSPHERE, EAU)
    assert avec_de_l_eau.metres == pytest.approx(10.33, abs=0.01)


def test_le_manometre_en_u_de_la_planche_13() -> None:
    """`P_2 = P_atm + rho g h`, la branche ouverte montant de `h`."""
    manometre = Manometre((Troncon(MERCURE, Hauteur(metres=0.25), Sens.MONTEE),))
    gaz = manometre.pression_amont(aval=Pression(pascals=ATMOSPHERE))
    colonne = variation_de_pression(MERCURE, Hauteur(metres=0.25))
    assert gaz.pascals == pytest.approx(ATMOSPHERE + colonne.pascals)


def test_les_couches_empilees_de_la_planche_14() -> None:
    """`P_atm + rho_1 g h_1 + rho_2 g h_2 + rho_3 g h_3 = P_1`."""
    couches = (
        Troncon(HUILE, Hauteur(metres=0.3), Sens.DESCENTE),
        Troncon(EAU, Hauteur(metres=0.2), Sens.DESCENTE),
        Troncon(MERCURE, Hauteur(metres=0.1), Sens.DESCENTE),
    )
    fond = Manometre(couches).pression_aval(amont=Pression(pascals=ATMOSPHERE))
    attendue = ATMOSPHERE + 9.81 * (850 * 0.3 + 1000 * 0.2 + 13_600 * 0.1)
    assert fond.pascals == pytest.approx(attendue)


def test_le_manometre_differentiel_de_la_planche_15() -> None:
    """Planche 15 : `P_1 - P_2 = (rho_2 - rho_1) g h`, le retrait `a` s'eliminant."""
    retrait, denivelee = 0.4, 0.15
    manometre = Manometre(
        (
            Troncon(EAU, Hauteur(metres=retrait + denivelee), Sens.DESCENTE),
            Troncon(MERCURE, Hauteur(metres=denivelee), Sens.MONTEE),
            Troncon(EAU, Hauteur(metres=retrait), Sens.MONTEE),
        )
    )
    amont = Pression(pascals=3e5)
    perte = amont.pascals - manometre.pression_aval(amont=amont).pascals
    assert perte == pytest.approx((13_600 - 1000) * 9.81 * denivelee)


def test_le_manometre_multi_fluides_de_la_planche_19_donne_130_kpa() -> None:
    """`P_1 + rho_w g h_1 + rho_oil g h_2 - rho_mer g h_3 = P_2`, a 85,6 kPa."""
    manometre = Manometre(
        (
            Troncon(EAU, Hauteur(metres=0.10), Sens.DESCENTE),
            Troncon(HUILE, Hauteur(metres=0.20), Sens.DESCENTE),
            Troncon(MERCURE, Hauteur(metres=0.35), Sens.MONTEE),
        )
    )
    air = manometre.pression_amont(aval=Pression(pascals=85_600.0))
    assert air.kilopascals == pytest.approx(130.0, abs=0.5)


PORTE = PorteCylindrique(rayon=0.8, niveau=5.0, masse_volumique=EAU)
"""Planche 19 : un cylindre de 0,8 m de rayon, l'eau a 5 m, par metre de long."""


def test_les_composantes_de_la_poussee_sur_la_porte() -> None:
    """Horizontale : `rho g h_C A`. Verticale : le fond, moins le bloc d'eau."""
    assert PORTE.poussee_horizontale() == pytest.approx(1000 * 9.81 * 4.6 * 0.8)
    sur_le_fond = 1000 * 9.81 * 5.0 * 0.8
    bloc = 1000 * 9.81 * 0.8**2 * (1 - math.pi / 4)
    assert PORTE.poussee_verticale() == pytest.approx(sur_le_fond - bloc)


def test_la_resultante_et_sa_ligne_d_action() -> None:
    """52,3 kN par metre, inclines de 46,4 degres, par le centre du cylindre."""
    assert PORTE.resultante() == pytest.approx(52.3e3, rel=2e-3)
    assert math.degrees(PORTE.inclinaison()) == pytest.approx(46.4, abs=0.1)


def test_le_poids_du_cylindre_equilibre_la_poussee_verticale() -> None:
    """Moments autour de la charniere : `F_R R sin(theta) = W R`, soit 37,9 kN/m."""
    assert PORTE.poids_a_l_ouverture() == pytest.approx(37.9e3, rel=2e-3)
    assert PORTE.poids_a_l_ouverture() == pytest.approx(
        PORTE.resultante() * math.sin(PORTE.inclinaison())
    )


def test_l_integrale_de_la_pression_sur_l_arc_redonne_les_deux_composantes() -> None:
    """Un second chemin : `dF = p R d(phi)`, normal a la paroi, donc vers le centre.

    L'arc mouille va de la charniere (a gauche du centre) au point bas.
    """
    rayon, niveau, pas = 0.8, 5.0, 20_000
    horizontale = verticale = 0.0
    for k in range(pas):
        angle = (k + 0.5) * (math.pi / 2) / pas
        profondeur = niveau - rayon + rayon * math.sin(angle)
        force = 1000 * 9.81 * profondeur * rayon * (math.pi / 2) / pas
        horizontale += force * math.cos(angle)
        verticale += force * math.sin(angle)
    assert horizontale == pytest.approx(PORTE.poussee_horizontale(), rel=1e-6)
    assert verticale == pytest.approx(PORTE.poussee_verticale(), rel=1e-6)


def test_une_porte_qui_depasse_de_l_eau_est_refusee() -> None:
    with pytest.raises(ValueError, match="niveau inferieur au rayon"):
        PorteCylindrique(rayon=0.8, niveau=0.5, masse_volumique=EAU)
    with pytest.raises(ValueError, match="rayon nul ou negatif"):
        PorteCylindrique(rayon=0.0, niveau=5.0, masse_volumique=EAU)
