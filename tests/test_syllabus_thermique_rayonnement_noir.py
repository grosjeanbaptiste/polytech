"""Le syllabus de thermique, tome 1 : l'echange direct entre surfaces noires.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_1.pdf`, pages 119 a
135 (deuxieme partie, chapitre 2), lues sur les pages rendues.

Les oracles : les equations (2.3) a (2.11), les formules des figures 2.5,
2.6, 2.9 et 2.10 -- recalculees par l'integrale (2.8) ou (2.9) --, les
courbes 1 et 2 de la figure 2.15, et les exercices II.2.1 et II.2.2.
"""

from __future__ import annotations

import math

import pytest
from scipy.integrate import dblquad, nquad

from polytech.thermique.domain.rayonnement.cordes_croisees import (
    aire_par_les_cordes,
    corde_tendue,
)
from polytech.thermique.domain.rayonnement.ecran_de_tubes import (
    aire_entre_deux_tubes,
    facteur_d_un_tube_vers_le_plan,
    facteur_direct_vers_la_seconde_rangee,
    facteur_du_plan_vers_la_rangee,
)
from polytech.thermique.domain.rayonnement.enceinte_noire import (
    aire_dans_une_cavite_a_trois,
    flux_entre_surfaces_noires,
    flux_nets_noirs,
    temperature_d_equilibre,
)
from polytech.thermique.domain.rayonnement.facteurs_analytiques import (
    element_vers_rectangle_parallele,
    element_vers_rectangle_perpendiculaire,
    rectangles_paralleles,
    rectangles_perpendiculaires,
)
from polytech.thermique.domain.rayonnement.facteurs_de_forme import (
    Enceinte,
    aire_d_echange_direct,
    facteur_entre_deux_plaques_paralleles,
)

SIGMA = 5.67e-8
"""La valeur que le syllabus emploie dans ses applications."""


def test_l_echange_entre_deux_surfaces_noires() -> None:
    """(2.3) : `Phi = 12 sigma (T_1^4 - T_2^4)`, nul a l'equilibre."""
    flux = flux_entre_surfaces_noires(2.0, 600.0, 300.0)
    assert flux == pytest.approx(2.0 * SIGMA * (600.0**4 - 300.0**4), rel=1e-3)
    assert flux_entre_surfaces_noires(2.0, 450.0, 450.0) == 0.0
    assert flux_entre_surfaces_noires(2.0, 300.0, 600.0) == pytest.approx(-flux)
    with pytest.raises(ValueError, match="aire d'echange negative"):
        flux_entre_surfaces_noires(-1.0, 600.0, 300.0)


def test_la_reciprocite_et_la_sommation() -> None:
    """(2.4) a (2.6) : `12 = F_12 S_1 = F_21 S_2`, `somme F_1j = 1`."""
    enceinte = Enceinte(
        surfaces=(2.0, 4.0, 6.0),
        facteurs={(0, 1): 0.4, (0, 2): 0.6, (1, 0): 0.2},
    )
    assert enceinte.reciproque(0, 1)
    assert enceinte.est_fermee_depuis(0)
    assert aire_d_echange_direct(2.0, 0.4) == pytest.approx(
        aire_d_echange_direct(4.0, 0.2)
    )


def test_l_element_face_a_un_rectangle_parallele() -> None:
    """Fig. 2.5, contre l'integrale (2.8) : `cos cos / (pi r^2)` sur le rectangle."""
    distance, largeur, longueur = 1.0, 0.7, 1.9

    def noyau(y: float, x: float) -> float:
        return distance**2 / (math.pi * (x**2 + y**2 + distance**2) ** 2)

    integre, _ = dblquad(noyau, 0.0, largeur, 0.0, longueur)
    facteur = element_vers_rectangle_parallele(largeur / distance, longueur / distance)
    assert facteur == pytest.approx(integre, rel=1e-9)
    assert facteur == pytest.approx(
        element_vers_rectangle_parallele(longueur / distance, largeur / distance)
    )
    assert element_vers_rectangle_parallele(1e6, 1e6) == pytest.approx(0.25, abs=1e-5)


def test_l_element_face_a_un_rectangle_perpendiculaire() -> None:
    """Fig. 2.6, contre l'integrale (2.8) ; `B = b / a`, `C = c / a`."""
    distance, largeur, hauteur = 1.0, 0.7, 1.9

    def noyau(z: float, x: float) -> float:
        return z * distance / (math.pi * (x**2 + z**2 + distance**2) ** 2)

    integre, _ = dblquad(noyau, 0.0, largeur, 0.0, hauteur)
    facteur = element_vers_rectangle_perpendiculaire(
        largeur / distance, hauteur / distance
    )
    assert facteur == pytest.approx(integre, rel=1e-9)
    assert facteur != pytest.approx(
        element_vers_rectangle_perpendiculaire(hauteur / distance, largeur / distance)
    )


def test_la_decomposition_de_la_figure_2_7() -> None:
    """Un element sous le milieu d'un cote : `F_12 = F_12' + F_12''`."""
    distance, demi_largeur, longueur = 2.0, 1.5, 4.0

    def noyau(y: float, x: float) -> float:
        return distance**2 / (math.pi * (x**2 + y**2 + distance**2) ** 2)

    integre, _ = dblquad(noyau, -demi_largeur, demi_largeur, 0.0, longueur)
    moitie = element_vers_rectangle_parallele(demi_largeur / 2.0, longueur / 2.0)
    assert 2 * moitie == pytest.approx(integre, rel=1e-9)


def test_les_rectangles_paralleles_de_la_figure_2_9() -> None:
    """La formule, contre l'integrale quadruple (2.9) et contre l'abaque."""
    largeur, longueur, distance = 2.0, 1.0, 1.0

    def noyau(x1: float, y1: float, x2: float, y2: float) -> float:
        carre = (x1 - x2) ** 2 + (y1 - y2) ** 2 + distance**2
        return distance**2 / (math.pi * carre**2)

    bornes = [[0, largeur], [0, longueur], [0, largeur], [0, longueur]]
    integre, _ = nquad(noyau, bornes, opts={"epsabs": 1e-6})
    facteur = rectangles_paralleles(largeur / distance, longueur / distance)
    assert facteur == pytest.approx(integre / (largeur * longueur), abs=1e-5)
    assert rectangles_paralleles(10.0, 10.0) == pytest.approx(0.83, abs=0.01)
    assert rectangles_paralleles(1.0, 0.1) == pytest.approx(0.025, abs=0.003)
    assert rectangles_paralleles(1.0, 1.0) == pytest.approx(0.20, abs=0.005)
    assert rectangles_paralleles(20.0, 0.1) == pytest.approx(0.05, abs=0.005)


def test_deux_carres_face_a_face_sont_un_cas_des_rectangles() -> None:
    """Le lien avec la formule des carres deja portee par la bibliotheque."""
    for rapport in (0.3, 1.0, 4.0):
        carres = facteur_entre_deux_plaques_paralleles(rapport, 1.0)
        assert rectangles_paralleles(rapport, rapport) == pytest.approx(carres)
    with pytest.raises(ValueError, match="rapport nul ou negatif"):
        rectangles_paralleles(0.0, 1.0)


def test_les_rectangles_perpendiculaires_de_la_figure_2_10() -> None:
    """`B = Y / X`, `C = Z / X` : 0,200 pour deux carres, et la reciprocite."""
    assert rectangles_perpendiculaires(1.0, 1.0) == pytest.approx(0.2000, abs=5e-5)
    assert rectangles_perpendiculaires(0.1, 10.0) == pytest.approx(0.44, abs=0.005)
    aller = 2.0 * rectangles_perpendiculaires(2.0, 1.0)
    retour = 1.0 * rectangles_perpendiculaires(1.0, 2.0)
    assert aller == pytest.approx(retour)
    with pytest.raises(ValueError, match="rapport nul ou negatif"):
        rectangles_perpendiculaires(1.0, -2.0)


def test_un_coin_de_cube_ferme_son_bilan() -> None:
    """Dans un cube : quatre faces voisines a 0,2 et la face opposee."""
    voisines = 4 * rectangles_perpendiculaires(1.0, 1.0)
    opposee = rectangles_paralleles(1.0, 1.0)
    assert voisines + opposee == pytest.approx(1.0, abs=1e-9)


def test_la_decomposition_de_la_figure_2_8() -> None:
    """`12 = (1+1')(2+2') + 1'2' - (1+1')2' - 1'(2+2')`, sur une arete d'un metre."""

    def aire(profondeur: float, hauteur: float) -> float:
        return profondeur * rectangles_perpendiculaires(profondeur, hauteur)

    proche, loin, bas, haut = 0.5, 1.5, 0.4, 1.0
    decomposee = aire(loin, haut) + aire(proche, bas) - aire(loin, bas)
    decomposee -= aire(proche, haut)

    def noyau(y: float, z: float, x1: float, x2: float) -> float:
        carre = (x1 - x2) ** 2 + y**2 + z**2
        return y * z / (math.pi * carre**2)

    bornes = [[proche, loin], [bas, haut], [0, 1], [0, 1]]
    integre, _ = nquad(noyau, bornes, opts={"epsabs": 1e-7})
    assert decomposee == pytest.approx(integre, abs=1e-5)


def test_la_cavite_a_trois_surfaces() -> None:
    """(2.10) : `12 = (S_1 + S_2 - S_3) / 2`, et les trois bilans se ferment."""
    cotes = (3.0, 4.0, 5.0)
    douze = aire_dans_une_cavite_a_trois(*cotes)
    treize = aire_dans_une_cavite_a_trois(cotes[0], cotes[2], cotes[1])
    vingt_trois = aire_dans_une_cavite_a_trois(cotes[1], cotes[2], cotes[0])
    assert (douze, treize, vingt_trois) == (1.0, 2.0, 3.0)
    assert douze + treize == cotes[0]
    assert douze + vingt_trois == cotes[1]
    assert treize + vingt_trois == cotes[2]
    with pytest.raises(ValueError, match="ne ferment pas une cavite"):
        aire_dans_une_cavite_a_trois(1.0, 1.0, 5.0)


def test_la_regle_des_elastiques_retrouve_la_cavite_a_trois() -> None:
    """(2.11) : sur un triangle, `AD` est nul et la regle redonne (2.10)."""
    a, b, c = (0.0, 0.0), (3.0, 0.0), (0.0, 4.0)
    croisees = (corde_tendue(a, a), corde_tendue(b, c))
    non_croisees = (corde_tendue(a, b), corde_tendue(a, c))
    with pytest.raises(ValueError, match="aire negative"):
        aire_par_les_cordes(croisees, non_croisees)
    aire = aire_par_les_cordes(non_croisees, croisees)
    assert aire == aire_dans_une_cavite_a_trois(3.0, 4.0, 5.0)


def test_l_aire_d_echange_entre_deux_tubes() -> None:
    """Page 131 : un segment croise moins un segment non croise.

    Le syllabus imprime un prefacteur `2 d_t` ; les cordes donnent `d_t`,
    et sa propre figure 2.15 le confirme (test suivant).
    """
    diametre, pas = 0.05, 0.1
    rayon, angle = diametre / 2, math.asin(diametre / pas)
    croisee = math.sqrt(pas**2 - diametre**2) + 2 * rayon * angle
    attendue = aire_par_les_cordes((croisee, croisee), (pas, pas))
    assert aire_entre_deux_tubes(diametre, pas) == pytest.approx(attendue)
    rapport = pas / diametre
    crochet = math.sqrt(rapport**2 - 1) - rapport + math.pi / 2 - math.acos(1 / rapport)
    assert aire_entre_deux_tubes(diametre, pas) == pytest.approx(diametre * crochet)
    imprimee = aire_entre_deux_tubes(diametre, pas, prefacteur=2.0)
    assert aire_entre_deux_tubes(diametre, pas) != pytest.approx(imprimee)
    assert aire_entre_deux_tubes(diametre, pas) == pytest.approx(0.01278, abs=5e-6)
    assert aire_entre_deux_tubes(diametre, pas) / diametre == pytest.approx(
        0.2556, abs=5e-5
    )
    assert imprimee == pytest.approx(0.02556, abs=1e-5)
    # F_PT du courrier : 0,66 (cordes) contre 0,53 (prefacteur 2 d), p = 2.
    assert facteur_du_plan_vers_la_rangee(diametre, pas) == pytest.approx(
        0.66, abs=0.005
    )
    assert facteur_du_plan_vers_la_rangee(
        diametre, pas, prefacteur=2.0
    ) == pytest.approx(0.53, abs=0.005)
    with pytest.raises(ValueError, match="tubes qui se chevauchent"):
        aire_entre_deux_tubes(0.05, 0.04)


@pytest.mark.parametrize(
    ("rapport", "lue"), [(2.0, 0.66), (3.0, 0.47), (4.0, 0.36), (7.0, 0.21)]
)
def test_la_courbe_1_de_la_figure_2_15(rapport: float, lue: float) -> None:
    """`F_PT = (pi d / 2 l)(1 - 2 F_12)`, lu sur l'abaque a 0,01 pres."""
    facteur = facteur_du_plan_vers_la_rangee(1.0, rapport)
    assert facteur == pytest.approx(lue, abs=0.01)
    avec_le_double = facteur_du_plan_vers_la_rangee(1.0, rapport, prefacteur=2.0)
    assert abs(avec_le_double - lue) > 0.005


def test_des_tubes_jointifs_arretent_tout() -> None:
    """A `l = d` l'ecran est un mur : `F_PT = 1` ; le facteur imprime rendrait 0,43."""
    assert facteur_du_plan_vers_la_rangee(1.0, 1.0) == pytest.approx(1.0)
    assert facteur_d_un_tube_vers_le_plan(1.0, 1.0) == pytest.approx(1 / math.pi)
    assert facteur_du_plan_vers_la_rangee(1.0, 1e4) == pytest.approx(0.0, abs=1e-3)


def test_le_bilan_d_un_tube_de_l_ecran() -> None:
    """`2 F_12 + F_1P + F_1P' = 1`, puis la reciprocite `S_P F_PT = S_T F_TP`."""
    diametre, pas = 0.05, 0.12
    vers_le_voisin = aire_entre_deux_tubes(diametre, pas) / (math.pi * diametre)
    vers_le_plan = facteur_d_un_tube_vers_le_plan(diametre, pas)
    assert 2 * vers_le_voisin + 2 * vers_le_plan == pytest.approx(1.0)
    depuis_le_plan = facteur_du_plan_vers_la_rangee(diametre, pas)
    assert pas * depuis_le_plan == pytest.approx(math.pi * diametre * vers_le_plan)


@pytest.mark.parametrize(("rapport", "lue"), [(2.0, 0.22), (4.0, 0.21), (7.0, 0.16)])
def test_la_courbe_2_de_la_figure_2_15(rapport: float, lue: float) -> None:
    """La seconde rangee recoit ce que la premiere laisse passer : `F (1 - F)`."""
    facteur = facteur_direct_vers_la_seconde_rangee(1.0, rapport)
    premiere = facteur_du_plan_vers_la_rangee(1.0, rapport)
    assert facteur == pytest.approx(premiere * (1 - premiere))
    assert facteur == pytest.approx(lue, abs=0.025)


def test_le_schema_equivalent_d_une_enceinte_noire() -> None:
    """2.4 : `Phi_i,net = somme ij sigma (T_i^4 - T_j^4)` ; la somme est nulle."""
    aires = ((0.0, 1.0, 2.0), (1.0, 0.0, 3.0), (2.0, 3.0, 0.0))
    temperatures = (800.0, 500.0, 300.0)
    flux = flux_nets_noirs(aires, temperatures)
    premier = flux_entre_surfaces_noires(1.0, 800.0, 500.0)
    premier += flux_entre_surfaces_noires(2.0, 800.0, 300.0)
    assert flux[0] == pytest.approx(premier)
    assert sum(flux) == pytest.approx(0.0, abs=1e-9)
    assert flux[0] > 0 > flux[2]
    with pytest.raises(ValueError, match="aires d'echange non symetriques"):
        flux_nets_noirs(((0.0, 1.0), (2.0, 0.0)), (800.0, 500.0))


def test_l_exercice_ii_2_1_deux_plaques_noires() -> None:
    """1,5 x 3 m a 1,5 m, 500 et 250 degres : `F = 0,285`, 20 481 W.

    Le syllabus ecrit 524 K pour 250 degres ; avec 523,15 K et le facteur
    non arrondi, 20 601 W.
    """
    facteur = rectangles_paralleles(3.0 / 1.5, 1.5 / 1.5)
    assert facteur == pytest.approx(0.285, abs=0.001)
    imprime = flux_entre_surfaces_noires(1.5 * 3.0 * 0.285, 773.0, 524.0)
    assert imprime == pytest.approx(20482.2, abs=0.1)
    assert imprime == pytest.approx(20481.0, abs=1.5)  # le syllabus arrondit
    flux = flux_entre_surfaces_noires(1.5 * 3.0 * facteur, 773.15, 523.15)
    assert flux == pytest.approx(20601.0, abs=0.5)


def test_l_exercice_ii_2_2_les_resistances_en_rangee() -> None:
    """25 x 2,5 mm, 10 mm d'intervalle, 500 A : 1385 K en rangee, 1091 K seule."""
    epaisseur, hauteur, intervalle = 2.5e-3, 25e-3, 10e-3
    lineique = 1.1e-6 / (hauteur * epaisseur)
    assert lineique == pytest.approx(0.0176)
    dissipe = lineique * 500.0**2
    assert dissipe == pytest.approx(4400.0)
    diagonale = corde_tendue((0.0, 0.0), (intervalle, hauteur))
    entre_voisines = aire_par_les_cordes((diagonale, diagonale), (intervalle,) * 2)
    perimetre = 2 * (epaisseur + hauteur)
    vers_l_ambiance = perimetre - 2 * entre_voisines
    assert vers_l_ambiance == pytest.approx(0.02115, abs=5e-6)
    en_rangee = temperature_d_equilibre(dissipe, vers_l_ambiance, 293.15)
    assert en_rangee == pytest.approx(1385.0, abs=1.0)
    seule = temperature_d_equilibre(dissipe, perimetre, 293.15)
    assert seule == pytest.approx(1091.0, abs=1.0)
    assert en_rangee > seule


def test_a_un_pas_de_deux_diametres_le_prefacteur_d_donne_0_66_et_2d_donne_0_53() -> (
    None
):
    rapport = 2.0
    assert facteur_du_plan_vers_la_rangee(1.0, rapport) == pytest.approx(0.66, abs=0.01)
    avec_le_double = facteur_du_plan_vers_la_rangee(1.0, rapport, prefacteur=2.0)
    assert avec_le_double == pytest.approx(0.53, abs=0.005)
