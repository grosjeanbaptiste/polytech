"""Planches de mecanique des fluides, chapitre IV.5 : l'equation de Bernoulli.

Source : `MecaniqueFluides/SLIDES/Slides_03_Equations_fondamentales.pdf`,
planches 60 a 72, lues sur les pages rendues.

Les oracles du cours : le reservoir de 5 m de la planche 65, les facteurs
de correction `alpha = 1` et `alpha = 2` de la planche 67, et la centrale
de la planche 69 (100 m^3/s, 120 m, 35 m de pertes, 80 %). Les exercices
des planches 71 et 72 sont donnes sans reponse : ceux dont l'enonce porte
toutes ses donnees sont resolus ici.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.fluides.domain.fondamentales.bernoulli import (
    MASSE_VOLUMIQUE,
    charge,
    vitesse_de_pitot,
    vitesse_de_torricelli,
)
from polytech.fluides.domain.fondamentales.bernoulli_generalise import (
    facteur_de_correction_cinetique,
    hauteur_manometrique,
    pression_d_arret,
    puissance_a_l_arbre_d_une_pompe,
    puissance_d_une_turbine,
    puissance_hydraulique,
    rendement,
)
from polytech.fluides.domain.fondamentales.continuite import vitesse_dans_la_section
from polytech.fluides.domain.fondamentales.remplissage import (
    duree_de_remplissage,
    hauteur_maximale_de_remplissage,
)
from polytech.fluides.domain.grandeurs import PESANTEUR
from polytech.fluides.domain.regimes import est_incompressible, nombre_de_mach

EAU = 1000.0


def test_la_pression_totale_se_conserve_le_long_d_une_ligne_de_courant() -> None:
    """Planche 60 : `rho V^2 / 2 + p + rho g z = cste = p_tot`."""
    en_haut = charge(
        pression=sp.Integer(0), altitude=sp.Integer(5), vitesse=sp.Integer(0)
    )
    vitesse = vitesse_de_torricelli(sp.Integer(5))
    en_bas = charge(pression=sp.Integer(0), altitude=sp.Integer(0), vitesse=vitesse)
    assert sp.simplify(en_haut - en_bas) == 0


def test_le_reservoir_de_cinq_metres_se_vide_a_dix_metres_par_seconde() -> None:
    """Planche 65 : `v_2 = racine(2 g z_1)`."""
    assert float(vitesse_de_torricelli(sp.Integer(5))) == pytest.approx(
        math.sqrt(2 * PESANTEUR * 5)
    )
    assert float(vitesse_de_torricelli(sp.Integer(5))) == pytest.approx(9.90, abs=0.01)


def test_la_pression_d_arret_ajoute_la_pression_dynamique() -> None:
    """Planche 64 : sur la ligne de courant `infini - A`, `v_A = 0`.

    La planche encadre `p_A = p_inf + v_inf^2 / 2` ; la ligne qui precede
    porte `p / rho`, et la masse volumique doit rester : `rho v^2 / 2`.
    """
    assert pression_d_arret(100_000.0, 1.2, 50.0) == pytest.approx(101_500.0)
    assert pression_d_arret(100_000.0, 1.2, 50.0) != pytest.approx(
        100_000.0 + 50.0**2 / 2
    )


def test_le_tube_de_pitot_mesure_la_vitesse_par_l_ecart_des_deux_prises() -> None:
    """Planche 64 : `p_A` a la prise d'impact, `p_B = p_inf` a la prise statique.

    La planche encadre `v_inf = racine(2 (p_B - p_A) / rho)` ; `p_A` est la
    plus grande des deux, et c'est `p_A - p_B` qui est sous la racine.
    """
    arret = pression_d_arret(100_000.0, 1.2, 50.0)
    ecart = sp.Float(arret - 100_000.0)
    lue = vitesse_de_pitot(ecart).subs(MASSE_VOLUMIQUE, 1.2)
    assert float(lue) == pytest.approx(50.0)
    assert arret > 100_000.0


def test_une_masse_volumique_nulle_n_a_pas_de_pression_dynamique() -> None:
    with pytest.raises(ValueError, match="masse volumique nulle ou negative"):
        pression_d_arret(100_000.0, 0.0, 50.0)


def test_bernoulli_ne_vaut_qu_en_incompressible() -> None:
    """Planche 62 : « liquide, ou gaz avec Ma < 0,3 ».

    Exercice 2 de la planche 71 : a 200 km/h l'equation s'applique, a
    1 050 km/h elle ne s'applique plus -- avec une celerite de 295 m/s a
    12 000 m, valeur de l'atmosphere standard, que l'enonce ne donne pas.
    """
    celerite = 295.0
    assert est_incompressible(nombre_de_mach(200 / 3.6, celerite))
    assert not est_incompressible(nombre_de_mach(1050 / 3.6, celerite))


RAYON = sp.Symbol("r", nonnegative=True)
PAROI = sp.Symbol("R", positive=True)


def test_le_facteur_de_correction_vaut_un_pour_un_profil_plat() -> None:
    """Planche 67 : ecoulements turbulents, profil plat, `alpha ~ 1`."""
    assert facteur_de_correction_cinetique(sp.Integer(3), RAYON, PAROI) == 1


def test_le_facteur_de_correction_vaut_deux_pour_un_profil_parabolique() -> None:
    """Planche 67 : ecoulements laminaires, `alpha = <v^3> / <v>^3 = 2`."""
    parabole = 1 - RAYON**2 / PAROI**2
    assert facteur_de_correction_cinetique(parabole, RAYON, PAROI) == 2


def test_un_profil_en_puissance_un_septieme_reste_proche_de_un() -> None:
    """Un profil turbulent usuel, `(1 - r / R)^(1/7)` : `alpha = 1,06`."""
    turbulent = (1 - RAYON / PAROI) ** sp.Rational(1, 7)
    alpha = facteur_de_correction_cinetique(turbulent, RAYON, PAROI)
    assert float(alpha) == pytest.approx(1.058, abs=1e-3)


def test_un_profil_de_debit_nul_n_a_pas_de_facteur() -> None:
    with pytest.raises(ValueError, match="vitesse moyenne nulle"):
        facteur_de_correction_cinetique(sp.Integer(0), RAYON, PAROI)


def test_la_centrale_de_la_planche_69_produit_67_megawatts() -> None:
    """100 m^3/s, 120 m de chute, 35 m de pertes, 80 % de rendement global."""
    electrique = puissance_d_une_turbine(
        EAU, debit=100.0, chute=120.0, pertes=35.0, rendement_global=0.80
    )
    assert electrique == pytest.approx(66.7e6, rel=2e-3)
    disponible = puissance_hydraulique(EAU, 100.0, 120.0)
    assert electrique == pytest.approx(disponible * (85 / 120) * 0.80)


def test_des_pertes_qui_depassent_la_chute_sont_refusees() -> None:
    with pytest.raises(ValueError, match="pertes superieures a la chute"):
        puissance_d_une_turbine(
            EAU, debit=100.0, chute=30.0, pertes=35.0, rendement_global=0.80
        )
    with pytest.raises(ValueError, match="rendement hors de"):
        puissance_d_une_turbine(
            EAU, debit=100.0, chute=120.0, pertes=35.0, rendement_global=1.2
        )


def test_l_exercice_4_de_la_planche_71() -> None:
    """70 m, 1 500 kg/s, 800 kW mecaniques, 750 kW electriques.

    La puissance disponible vaut `m g h = 1 030 kW` : la turbine rend
    77,7 %, l'ensemble 72,8 %.
    """
    disponible = puissance_hydraulique(EAU, 1500.0 / EAU, 70.0)
    assert disponible == pytest.approx(1030e3, rel=1e-3)
    assert rendement(800e3, disponible) == pytest.approx(0.777, abs=1e-3)
    assert rendement(750e3, disponible) == pytest.approx(0.728, abs=1e-3)
    assert rendement(750e3, 800e3) == pytest.approx(0.9375)


def test_un_rendement_ne_depasse_pas_l_unite() -> None:
    with pytest.raises(ValueError, match="plus que fourni"):
        rendement(900e3, 800e3)
    with pytest.raises(ValueError, match="puissance fournie nulle"):
        rendement(1.0, 0.0)


def test_l_exercice_7_de_la_planche_72_le_bateau_pompe() -> None:
    """1 030 kg/m^3, 0,1 m^3/s, buse de 5 cm a 4 m, 3 m de pertes, 70 %.

    L'eau sort a 50,9 m/s ; la pompe demande 201 kW a son arbre.
    """
    section = math.pi * 0.05**2 / 4
    ejection = vitesse_dans_la_section(0.1, section)
    assert ejection == pytest.approx(50.9, abs=0.05)
    hauteur = hauteur_manometrique(denivele=4.0, vitesse_finale=ejection, pertes=3.0)
    assert hauteur == pytest.approx(139.2, abs=0.1)
    arbre = puissance_a_l_arbre_d_une_pompe(1030.0, 0.1, hauteur, rendement_pompe=0.70)
    assert arbre == pytest.approx(201e3, rel=3e-3)


def test_la_hauteur_manometrique_compte_les_quatre_termes() -> None:
    """Planche 68 : altitude, energie cinetique, pression, pertes."""
    hauteur = hauteur_manometrique(
        denivele=2.0,
        vitesse_finale=4.0,
        vitesse_initiale=2.0,
        hauteur_de_pression=9810.0 / (EAU * PESANTEUR),
        pertes=1.5,
    )
    cinetique = (4.0**2 - 2.0**2) / (2 * PESANTEUR)
    assert hauteur == pytest.approx(2.0 + cinetique + 1.0 + 1.5)


def test_l_exercice_3_de_la_planche_71_la_ventilation() -> None:
    """200 m^2 sous 2,7 m, 35 % par heure : 3 150 L/min, et 10,6 cm a 6 m/s."""
    debit = 0.35 * 200.0 * 2.7 / 3600
    assert debit * 60_000 == pytest.approx(3150.0)
    diametre = math.sqrt(4 * debit / (math.pi * 6.0))
    assert diametre == pytest.approx(0.106, abs=5e-4)
    assert vitesse_dans_la_section(debit, math.pi * diametre**2 / 4) == pytest.approx(
        6.0
    )


def test_l_exercice_1_de_la_planche_71_la_hauteur_maximale() -> None:
    """Le niveau se stabilise quand l'orifice evacue le debit entrant."""
    debit, orifice = 5.0, 0.02
    maximale = hauteur_maximale_de_remplissage(debit, EAU, orifice)
    section = math.pi * orifice**2 / 4
    sortant = EAU * section * float(vitesse_de_torricelli(sp.Float(maximale)))
    assert sortant == pytest.approx(debit)


def test_l_exercice_1_de_la_planche_71_la_loi_du_niveau() -> None:
    """`rho A_T dz/dt = m_point - rho A_o racine(2 g z)`, integree en `t(z)`."""
    debit, masse, section, orifice = sp.symbols("m rho A_T A_o", positive=True)
    niveau = sp.Symbol("z", positive=True)
    duree = duree_de_remplissage(niveau, debit, masse, section, orifice)
    pente = 1 / sp.diff(duree, niveau)
    bilan = (debit - masse * orifice * sp.sqrt(2 * sp.Symbol("g") * niveau)) / (
        masse * section
    )
    ecart = sp.simplify(pente - bilan.subs(sp.Symbol("g"), sp.nsimplify(PESANTEUR)))
    assert ecart == 0
    assert sp.limit(duree, niveau, 0, "+") == 0


def test_le_niveau_met_un_temps_infini_a_atteindre_son_maximum() -> None:
    """L'approche est asymptotique : `t(z)` diverge en `z_max`."""
    debit, orifice, cuve = 5.0, math.pi * 0.02**2 / 4, 0.5
    maximale = hauteur_maximale_de_remplissage(debit, EAU, 0.02)
    proche = duree_de_remplissage(0.999 * maximale, debit, EAU, cuve, orifice)
    plus_proche = duree_de_remplissage(0.999999 * maximale, debit, EAU, cuve, orifice)
    assert float(plus_proche) > 1.5 * float(proche)


def test_les_gardes_de_la_pompe_et_du_reservoir() -> None:
    with pytest.raises(ValueError, match="rendement hors de"):
        puissance_a_l_arbre_d_une_pompe(EAU, 0.1, 10.0, rendement_pompe=1.3)
    with pytest.raises(ValueError, match="rendement hors de"):
        puissance_a_l_arbre_d_une_pompe(EAU, 0.1, 10.0, rendement_pompe=0.0)
    with pytest.raises(ValueError, match="nuls ou negatifs"):
        hauteur_maximale_de_remplissage(5.0, EAU, 0.0)
    with pytest.raises(ValueError, match="nuls ou negatifs"):
        hauteur_maximale_de_remplissage(5.0, 0.0, 0.02)
