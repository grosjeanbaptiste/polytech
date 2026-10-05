"""Le syllabus de thermique, tome 2 : les echangeurs de chaleur.

Source : `ThermiqueFondements/SYLLABUS/Syllabus_Tome_2_2013.pdf`, pages 115
a 135 (deuxieme partie, chapitre 2).

Les oracles : les relations (2.4) a (2.18), et les trois exercices --
568 814 W, 43,9 degres, 10,37 m^2 et 36 tubes ; `eta_1 = 0,578`, 68,3 et
69,3 degres ; 76,3 degres et 14 326 W.
"""

from __future__ import annotations

import math

import pytest
from scipy.integrate import solve_ivp

from polytech.thermique.domain.continuite_du_flux import temperature_superficielle
from polytech.thermique.domain.echangeur_a_coefficient_variable import (
    coefficient_moyen,
    flux_a_coefficient_lineaire,
)
from polytech.thermique.domain.echangeurs import (
    Echangeur,
    efficacite_a_contre_courant,
    efficacite_a_courants_paralleles,
    moyenne_logarithmique,
)
from polytech.thermique.domain.echangeurs_a_courants_paralleles import (
    Extremites,
    Sens,
    debit_condense,
    flux_du_bilan,
    surface_d_echange,
)
from polytech.thermique.domain.integration_d_echangeur import (
    pincement,
    surface_par_integration,
    temperature_de_sortie_par_integration,
)
from polytech.thermique.domain.rayonnement_equivalent import coefficient_du_gaz
from polytech.thermique.domain.rendement_d_echangeur import (
    efficacite,
    rendement,
    sortie_du_fluide_chaud,
    sortie_du_fluide_froid,
)
from polytech.thermique.domain.tables.eau_saturee import temperature_de_saturation

EXTREMITES = Extremites(
    entree_chaude=110.0, sortie_chaude=70.0, entree_froide=30.0, sortie_froide=60.0
)


def test_le_bilan_d_un_fluide_sans_changement_de_phase() -> None:
    """(2.6) : `Phi = q_m c (t_E - t_S)` ; exercice I, 568 814 W."""
    flux = flux_du_bilan(3.78, 4180.0, 74.0 - 38.0)
    assert flux == pytest.approx(568814.0, abs=1.0)
    with pytest.raises(ValueError, match="debit de capacite nul ou negatif"):
        flux_du_bilan(0.0, 4180.0, 36.0)


def test_le_bilan_d_un_condenseur() -> None:
    """(2.12) : `Phi = q_m,1 l_v` ; exercice I, 0,25 kg/s a 2250,8 kJ/kg."""
    assert debit_condense(568814.0, 2250.8e3) == pytest.approx(0.25, abs=0.005)
    with pytest.raises(ValueError, match="chaleur latente nulle ou negative"):
        debit_condense(568814.0, 0.0)


def test_les_ecarts_d_extremite_dependent_du_sens() -> None:
    """Figures 2.8 et 2.9 : memes entrees face a face, ou entree face a sortie."""
    assert EXTREMITES.ecarts(Sens.COCOURANT) == (80.0, 10.0)
    assert EXTREMITES.ecarts(Sens.CONTRECOURANT) == (50.0, 40.0)
    with pytest.raises(ValueError, match="le fluide chaud ne se refroidit pas"):
        Extremites(70.0, 110.0, 30.0, 60.0)
    with pytest.raises(ValueError, match="le fluide froid ne se rechauffe pas"):
        Extremites(110.0, 70.0, 60.0, 30.0)


def test_la_moyenne_logarithmique() -> None:
    """(2.9) et (2.10) : `(Dt_B - Dt_A) / ln(Dt_B / Dt_A)`."""
    cocourant = EXTREMITES.moyenne_logarithmique(Sens.COCOURANT)
    assert cocourant == pytest.approx(70.0 / math.log(8.0))
    assert cocourant == pytest.approx(moyenne_logarithmique(80.0, 10.0))
    contrecourant = EXTREMITES.moyenne_logarithmique(Sens.CONTRECOURANT)
    assert contrecourant == pytest.approx(10.0 / math.log(1.25))


def test_le_contrecourant_demande_moins_de_surface() -> None:
    """Section 2.5 : sa moyenne logarithmique est toujours la plus grande."""
    cocourant = EXTREMITES.moyenne_logarithmique(Sens.COCOURANT)
    contrecourant = EXTREMITES.moyenne_logarithmique(Sens.CONTRECOURANT)
    assert contrecourant > cocourant
    flux = 50000.0
    assert surface_d_echange(flux, 300.0, contrecourant) < surface_d_echange(
        flux, 300.0, cocourant
    )
    croise = Extremites(110.0, 50.0, 30.0, 60.0)
    assert croise.moyenne_logarithmique(Sens.CONTRECOURANT) > 0
    with pytest.raises(ValueError, match="signes opposes"):
        croise.moyenne_logarithmique(Sens.COCOURANT)


def test_le_condenseur_ne_distingue_pas_les_deux_sens() -> None:
    """(2.11) et exercice I : 102,3 degres a 1,1 bar, `Dt_log = 43,9`."""
    saturation = temperature_de_saturation(1.1e5)
    assert saturation == pytest.approx(102.3, abs=0.05)
    condenseur = Extremites(102.3, 102.3, 38.0, 74.0)
    attendu = (74.0 - 38.0) / math.log((102.3 - 38.0) / (102.3 - 74.0))
    for sens in Sens:
        assert condenseur.moyenne_logarithmique(sens) == pytest.approx(attendu)
    assert attendu == pytest.approx(43.9, abs=0.05)


def test_l_exercice_i_le_dimensionnement() -> None:
    """Page 132 : 10,37 m^2, 36 tubes.

    « L_t = 10,37 / (36 pi 0,01905) = 4,83 m » : le quotient vaut 4,81 ;
    4,83 s'obtient avec les 35,84 tubes d'avant l'arrondi.
    """
    surface = surface_d_echange(568814.0, 1250.0, 43.9)
    assert surface == pytest.approx(10.37, abs=0.005)
    tubes = 3.78 / (1000.0 * 0.37 * math.pi / 4 * 0.01905**2)
    assert tubes == pytest.approx(35.84, abs=0.01)
    assert math.ceil(tubes) == 36
    assert 10.37 / (36 * math.pi * 0.01905) == pytest.approx(4.81, abs=0.005)
    assert 10.37 / (tubes * math.pi * 0.01905) == pytest.approx(4.83, abs=0.005)
    with pytest.raises(ValueError, match="coefficient global nul ou negatif"):
        surface_d_echange(568814.0, 0.0, 43.9)


def test_les_rendements_des_deux_fluides() -> None:
    """(2.13) : l'ecart reel d'un fluide rapporte a `t_1E - t_2E`."""
    chaud, froid = EXTREMITES.rendements()
    assert chaud == pytest.approx(40.0 / 80.0)
    assert froid == pytest.approx(30.0 / 80.0)


@pytest.mark.parametrize("sens", list(Sens))
def test_le_rendement_ne_depend_que_de_n_et_de_r(sens: Sens) -> None:
    """(2.14) a (2.16) : elimination des temperatures, verifiee par le bilan."""
    unites, rapport = 1.3, 0.6
    eta = rendement(sens, unites, rapport)
    chaude = sortie_du_fluide_chaud(110.0, 30.0, eta)
    froide = sortie_du_fluide_froid(110.0, 30.0, rapport * eta)
    extremites = Extremites(110.0, chaude, 30.0, froide)
    capacite = 2000.0
    transfert = Echangeur(unites * capacite).flux(*extremites.ecarts(sens))
    assert transfert == pytest.approx(flux_du_bilan(capacite, 1.0, 110.0 - chaude))


def test_les_rendements_rejoignent_les_efficacites_du_formulaire() -> None:
    """Meme loi que `echangeurs`, mais pour tout `R_1`, zero et au-dela de un."""
    assert rendement(Sens.CONTRECOURANT, 1.3, 0.6) == pytest.approx(
        efficacite_a_contre_courant(1.3, 0.6)
    )
    assert rendement(Sens.COCOURANT, 1.3, 0.6) == pytest.approx(
        efficacite_a_courants_paralleles(1.3, 0.6)
    )
    assert rendement(Sens.CONTRECOURANT, 1.3, 1.0) == pytest.approx(1.3 / 2.3)
    assert rendement(Sens.CONTRECOURANT, 1.3, 1.6) < 1 / 1.6
    with pytest.raises(ValueError, match="nombre d'unites nul ou negatif"):
        rendement(Sens.COCOURANT, 0.0, 0.6)
    with pytest.raises(ValueError, match="rapport des debits de capacite negatif"):
        rendement(Sens.COCOURANT, 1.3, -0.1)


def test_le_condenseur_et_le_bouilleur() -> None:
    """Page 124 : `R_1 = 0` donne `1 - exp(-N_1)` dans les deux sens."""
    for sens in Sens:
        assert rendement(sens, 0.7, 0.0) == pytest.approx(1 - math.exp(-0.7))


def test_les_limites_d_un_echangeur_tres_grand() -> None:
    """Remarque 1 : a `R_1 = 1`, le contrecourant tend vers 1, le cocourant vers 0,5."""
    assert rendement(Sens.CONTRECOURANT, 1e6, 1.0) == pytest.approx(1.0, abs=1e-5)
    assert rendement(Sens.COCOURANT, 50.0, 1.0) == pytest.approx(0.5)
    for unites in (0.3, 1.0, 4.0):
        assert rendement(Sens.CONTRECOURANT, unites, 0.8) > rendement(
            Sens.COCOURANT, unites, 0.8
        )


def test_l_efficacite_se_rapporte_au_plus_petit_debit() -> None:
    """Remarque 3 : `epsilon` est le plus eleve des deux rendements."""
    conductance, chaude, froide = 4209.0, 3559.8, 4723.4
    epsilon = efficacite(Sens.CONTRECOURANT, conductance, (chaude, froide))
    eta_1 = rendement(Sens.CONTRECOURANT, conductance / chaude, chaude / froide)
    eta_2 = eta_1 * chaude / froide
    assert epsilon == pytest.approx(max(eta_1, eta_2))
    inverse = efficacite(Sens.CONTRECOURANT, conductance, (froide, chaude))
    assert inverse == pytest.approx(epsilon)


def test_l_exercice_2_l_huile_refroidie_a_contrecourant() -> None:
    """Page 133 : `N_1 = 1,18`, `R_1 = 0,754`, `eta_1 = 0,578`, 68,3 et 69,3 degres.

    La chaine imprimee part de `N_1` et `R_1` arrondis ; elle se
    reproduit ainsi, 148 557 W compris. Sans arrondi intermediaire,
    `eta_1 = 0,5786` et l'huile sort a 68,2 degres.
    """
    huile, eau = 1.89 * 1883.5, 1.13 * 4180.0
    unites, rapport = 300.0 * 14.03 / huile, huile / eau
    assert unites == pytest.approx(1.18, abs=0.005)
    assert rapport == pytest.approx(0.754, abs=0.0005)
    eta = rendement(Sens.CONTRECOURANT, 1.18, 0.754)
    assert eta == pytest.approx(0.578, abs=0.0005)
    chaude = sortie_du_fluide_chaud(110.0, 37.8, eta)
    assert chaude == pytest.approx(68.3, abs=0.05)
    flux = flux_du_bilan(1.89, 1883.5, 110.0 - chaude)
    assert flux == pytest.approx(148557.0, rel=2e-4)
    froide = sortie_du_fluide_froid(110.0, 37.8, 0.754 * eta)
    assert froide == pytest.approx(69.3, abs=0.05)
    assert froide == pytest.approx(37.8 + flux / (0.754**-1 * huile), rel=1e-9)
    assert froide > chaude
    exact = rendement(Sens.CONTRECOURANT, unites, rapport)
    assert exact == pytest.approx(0.5786, abs=0.0001)
    assert sortie_du_fluide_chaud(110.0, 37.8, exact) == pytest.approx(68.2, abs=0.05)


def test_l_exercice_3_l_huile_rechauffee_par_de_la_vapeur() -> None:
    """Page 134 : `S_i = 2,111 m^2`, `105 - t_2S = 28,7`, 76,3 degres, 14 326 W.

    Le syllabus ecrit « eta_2 = 1 - exp(-R_2) » : c'est `N_2`. Et la
    saturation a 1,2 bar est a 104,8 degres, arrondie a 105.
    """
    assert temperature_de_saturation(1.2e5) == pytest.approx(104.8, abs=0.05)
    surface = 24 * math.pi * 0.014 * 2.0
    assert surface == pytest.approx(2.111, abs=0.0005)
    unites = 160.1 * surface / (0.23 * 1990.0)
    eta = rendement(Sens.CONTRECOURANT, unites, 0.0)
    assert (105.0 - 45.0) * (1 - eta) == pytest.approx(28.7, abs=0.05)
    sortie = 45.0 + eta * (105.0 - 45.0)
    assert sortie == pytest.approx(76.3, abs=0.05)
    assert flux_du_bilan(0.23, 1990.0, 76.3 - 45.0) == pytest.approx(14326.0, abs=1)
    assert (45.0 + sortie) / 2 == pytest.approx(60.7, abs=0.05)


def test_la_caracteristique_de_fonctionnement_est_lineaire() -> None:
    """Remarque 2 : `t_1S = t_1E - eta_1 (t_1E - t_2E)`, affine en `t_2E`."""
    sorties = [sortie_du_fluide_chaud(110.0, froide, 0.578) for froide in (20, 30, 40)]
    assert sorties[1] - sorties[0] == pytest.approx(sorties[2] - sorties[1])
    assert sorties[1] - sorties[0] == pytest.approx(0.578 * 10)
    with pytest.raises(ValueError, match="rendement hors de"):
        sortie_du_fluide_chaud(110.0, 30.0, 1.2)


def test_un_coefficient_peu_variable_se_moyenne() -> None:
    """Section 2.4.3 : `K_gl,moy = (K_gl,A + K_gl,B) / 2`."""
    assert coefficient_moyen(280.0, 320.0) == pytest.approx(300.0)


def test_le_coefficient_lineaire_en_l_ecart() -> None:
    """(2.17) : `Phi = S (K_A Dt_B - K_B Dt_A) / ln(K_A Dt_B / (K_B Dt_A))`."""
    uniforme = flux_a_coefficient_lineaire(10.0, (300.0, 80.0), (300.0, 10.0))
    assert uniforme == pytest.approx(3000.0 * moyenne_logarithmique(80.0, 10.0))
    ordonnee, pente, pas = 250.0, 1.0, 1e-3

    def derivee(_: float, etat: list[float]) -> list[float]:
        return [-pas * (ordonnee + pente * etat[0]) * etat[0]]

    surface = 10.0
    integre = solve_ivp(derivee, (0.0, surface), [80.0], rtol=1e-10, atol=1e-12)
    finale = float(integre.y[0][-1])
    flux = flux_a_coefficient_lineaire(
        surface, (ordonnee + pente * 80.0, 80.0), (ordonnee + pente * finale, finale)
    )
    assert flux == pytest.approx((80.0 - finale) / pas, rel=1e-7)
    assert flux_a_coefficient_lineaire(
        surface, (330.0, 40.0), (330.0, 40.0)
    ) == pytest.approx(surface * 330.0 * 40.0)


CAPACITES = (2000.0, 3000.0)


def _froid_a_contrecourant(chaud: float) -> float:
    return 60.0 - CAPACITES[0] / CAPACITES[1] * (110.0 - chaud)


def _densite(chaud: float) -> float:
    return 300.0 * (chaud - _froid_a_contrecourant(chaud))


def test_l_integration_numerique_retrouve_la_moyenne_logarithmique() -> None:
    """(2.18) : `S = q_m,1 integrale(dh_1 / phi)`, ici a `K_gl` et `c_1` constants."""
    surface = surface_par_integration(CAPACITES[0], _densite, entree=110.0, sortie=65.0)
    extremites = Extremites(110.0, 65.0, _froid_a_contrecourant(65.0), 60.0)
    flux = flux_du_bilan(CAPACITES[0], 1.0, 45.0)
    attendue = surface_d_echange(
        flux, 300.0, extremites.moyenne_logarithmique(Sens.CONTRECOURANT)
    )
    assert surface == pytest.approx(attendue, rel=1e-9)
    sortie = temperature_de_sortie_par_integration(
        CAPACITES[0], _densite, entree=110.0, surface=attendue, plancher=31.0
    )
    assert sortie == pytest.approx(65.0, abs=1e-6)


def test_la_chaleur_massique_variable_s_integre_aussi() -> None:
    """Section 2.6.1 : la methode vaut quand `c_1` varie avec la temperature."""

    def capacite(chaud: float) -> float:
        return 1.0 + 0.002 * chaud

    surface = surface_par_integration(
        2000.0, lambda t: 300.0 * (t - 20.0), entree=110.0, sortie=65.0, pente=capacite
    )
    exacte = 2000.0 / 300.0 * (1.04 * math.log(2.0) + 0.002 * 45.0)
    assert surface == pytest.approx(exacte, rel=1e-9)


def test_le_point_de_pincement() -> None:
    """Figure 2.16 : l'ecart minimal entre les deux profils, qui doit rester positif."""
    chaud = (500.0, 380.0, 300.0, 200.0)
    froid = (420.0, 250.0, 250.0, 60.0)
    assert pincement(chaud, froid) == (2, 50.0)
    with pytest.raises(ValueError, match="les profils se croisent"):
        pincement((500.0, 240.0), (420.0, 250.0))
    with pytest.raises(ValueError, match="les profils se croisent"):
        pincement((500.0, 250.0), (420.0, 250.0))
    with pytest.raises(ValueError, match="profils de longueurs differentes"):
        pincement((500.0, 240.0), (420.0,))


SATURATION, RAYONS, ACIER, EBULLITION = 473.15, (0.030, 0.035), 40.0, 10000.0
"""Un tube de fumees : eau a 200 degres, acier de 5 mm, `K_eb` tres eleve."""


def _densite_des_fumees(fumees: float) -> float:
    """(2.19) : rayonnement du gaz et convection, egaux a ce que la paroi evacue."""
    interne, externe = RAYONS

    def recu(paroi: float) -> float:
        rayonnement = coefficient_du_gaz(
            0.8, emission=0.15, absorption=0.2, gaz=fumees, paroi=paroi
        )
        return (rayonnement + 30.0) * (fumees - paroi)

    def evacue(paroi: float) -> float:
        resistance = interne / ACIER * math.log(externe / interne)
        return (paroi - SATURATION) / (resistance + interne / (externe * EBULLITION))

    paroi = temperature_superficielle(recu, evacue, (SATURATION, fumees - 1.0))
    return evacue(paroi)


def test_l_application_au_tube_de_fumees() -> None:
    """Section 2.6.2 : `integrale(dh_1 / phi_i) = pi d_i L / q_m,1`.

    Le syllabus ne chiffre que les ordres de grandeur, 1500 et 850
    degres ; les autres donnees sont choisies ici, et seule la coherence
    de la methode est eprouvee : la surface integree, la sortie
    retrouvee, et la marche pas a pas de (2.18).
    """
    capacite, entree, sortie = 600.0, 1773.15, 1123.15
    surface = surface_par_integration(
        capacite, _densite_des_fumees, entree=entree, sortie=sortie
    )
    assert surface > 0
    retrouvee = temperature_de_sortie_par_integration(
        capacite, _densite_des_fumees, entree=entree, surface=surface, plancher=600.0
    )
    assert retrouvee == pytest.approx(sortie, abs=1e-4)
    marche = solve_ivp(
        lambda _, etat: [-_densite_des_fumees(etat[0]) / capacite],
        (0.0, surface),
        [entree],
        rtol=1e-9,
    )
    assert float(marche.y[0][-1]) == pytest.approx(sortie, abs=1e-3)
    assert _densite_des_fumees(entree) > 3 * _densite_des_fumees(sortie)
