"""Syllabus d'electricite generale, chapitre 10 : les milieux aimantes.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 10-2 a 10-18 et 10-29 a 10-30, lues sur les pages rendues.

Le tableau 10.3.1 est recopie tel quel et confronte au classement que le
texte en donne. Les courbes d'aimantation ne sont que dessinees : les tests
les eprouvent sur des cycles dont l'aire se calcule a la main -- un
rectangle, que la figure 10.9.1a idealise, et une ellipse.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.constantes import PERMEABILITE_DU_VIDE
from polytech.electromagnetisme.domain.energie_et_forces import (
    densite_d_energie_magnetique,
)
from polytech.electromagnetisme.domain.magnetostatique.biot_et_savart import (
    PERMEABILITE,
)
from polytech.electromagnetisme.domain.magnetostatique.champ_demagnetisant import (
    champ_demagnetisant,
    champ_interieur,
    part_du_champ_applique,
)
from polytech.electromagnetisme.domain.magnetostatique.champ_magnetique import (
    champ_magnetique,
)
from polytech.electromagnetisme.domain.magnetostatique.champ_terrestre import (
    COMPOSANTE_HORIZONTALE,
    COMPOSANTE_VERTICALE,
    inclinaison,
    intensite,
)
from polytech.electromagnetisme.domain.magnetostatique.circuits_magnetiques import (
    Magnetisme,
    classe_de_magnetisme,
)
from polytech.electromagnetisme.domain.magnetostatique.courbes_d_aimantation import (
    aire_du_cycle,
    champ_coercitif,
    induction_remanente,
    pente_asymptotique,
    permeabilite_differentielle,
    permeabilite_initiale,
    travail_volumique,
)
from polytech.electromagnetisme.domain.magnetostatique.milieux_aimantes import (
    CORPS_DIAMAGNETIQUES,
    CORPS_PARAMAGNETIQUES,
    TEMPERATURE_DE_CURIE_DU_FER,
    aimantation_induite,
    est_ferromagnetique,
    induction_du_milieu,
    susceptibilite,
)

H = sp.Symbol("H", positive=True)


def test_le_tableau_10_3_1_range_chaque_corps_sous_son_titre() -> None:
    """Page 10-5 : `mu_r` inferieur a l'unite, ou superieur."""
    assert CORPS_DIAMAGNETIQUES == {
        "cuivre": 0.999990,
        "eau": 0.999991,
        "bismuth": 0.99984,
    }
    assert CORPS_PARAMAGNETIQUES == {
        "aluminium": 1.000020,
        "air": 1.0000004,
        "oxygene": 1.000002,
        "FeCl3": 1.003,
    }
    for relative in CORPS_DIAMAGNETIQUES.values():
        assert classe_de_magnetisme(relative) is Magnetisme.DIAMAGNETIQUE
    for relative in CORPS_PARAMAGNETIQUES.values():
        assert classe_de_magnetisme(relative) is Magnetisme.PARAMAGNETIQUE


def test_la_susceptibilite_est_la_permeabilite_relative_moins_un() -> None:
    """(10.3.6). Le cuivre : dix millioniemes, et negative."""
    assert susceptibilite(0.999990) == pytest.approx(-1.0e-5)
    assert susceptibilite(1.003) == pytest.approx(3.0e-3)


def test_une_permeabilite_relative_nulle_n_a_pas_de_susceptibilite() -> None:
    with pytest.raises(ValueError, match="permeabilite relative nulle ou negative"):
        susceptibilite(0.0)


def test_l_aimantation_diamagnetique_s_oppose_au_champ() -> None:
    """(10.3.5) et page 10-2 : `M_d` opposee, `M_p` dans le sens du champ."""
    assert aimantation_induite(CORPS_DIAMAGNETIQUES["bismuth"], 1000.0) < 0
    assert aimantation_induite(CORPS_PARAMAGNETIQUES["FeCl3"], 1000.0) == (
        pytest.approx(3.0)
    )


def test_les_deux_ecritures_de_l_induction_s_accordent() -> None:
    """(10.3.2) `B = mu_0 (H + M)` et (10.3.3) `B = mu H`, puis (8.16.2)."""
    relative, champ = 1.003, 250.0
    aimante = aimantation_induite(relative, champ)
    induction = induction_du_milieu(champ, aimante)
    assert induction == pytest.approx(relative * PERMEABILITE_DU_VIDE * champ)
    retour = champ_magnetique((induction, 0, 0), (aimante, 0, 0))[0]
    assert float(retour.subs(PERMEABILITE, PERMEABILITE_DU_VIDE)) == pytest.approx(
        champ
    )


def test_le_fer_est_ferromagnetique_sous_sa_temperature_de_curie() -> None:
    """Page 10-4 : « de l'ordre de 1 050 K »."""
    assert TEMPERATURE_DE_CURIE_DU_FER == 1050.0
    assert est_ferromagnetique(293.0, TEMPERATURE_DE_CURIE_DU_FER)
    assert not est_ferromagnetique(1100.0, TEMPERATURE_DE_CURIE_DU_FER)


def test_une_temperature_absolue_negative_est_refusee() -> None:
    with pytest.raises(ValueError, match="temperature absolue negative"):
        est_ferromagnetique(-1.0, TEMPERATURE_DE_CURIE_DU_FER)


def test_le_travail_volumique_d_un_milieu_lineaire_est_mu_h2_sur_deux() -> None:
    """(10.4.1) integree donne (10.4.2), que le chapitre 9 ecrivait deja."""
    permeabilite = 500 * PERMEABILITE_DU_VIDE
    travail = travail_volumique(permeabilite * H, H, 0, 80)
    assert float(travail) == pytest.approx(
        densite_d_energie_magnetique(permeabilite, 80)
    )


def test_la_courbe_de_premiere_aimantation_a_une_asymptote_de_pente_mu0() -> None:
    """Page 10-8 : `M` tend vers `M_s`, donc `B` vers une oblique de pente `mu_0`.

    La courbe d'essai est choisie par le test, pas par le cours : une
    aimantation `M_s H / (H + a)` qui sature.
    """
    mu0, saturation, demi = sp.symbols("mu_0 M_s a", positive=True)
    induction = mu0 * (H + saturation * H / (H + demi))
    assert pente_asymptotique(induction, H) == mu0
    assert permeabilite_initiale(induction, H) == mu0 * (1 + saturation / demi)
    differentielle = permeabilite_differentielle(induction, H)
    assert sp.simplify(differentielle.subs(H, 0) - mu0 * (1 + saturation / demi)) == 0
    assert sp.limit(differentielle, H, sp.oo) == mu0


CYCLE_RECTANGULAIRE = ((-40.0, -1.2), (40.0, -1.2), (40.0, 1.2), (-40.0, 1.2))
"""Figure 10.9.1a idealisee : `H_c = 40 A/m`, `B_s = 1,2 T`, sens du parcours."""


def test_l_aire_du_cycle_rectangulaire_est_quatre_fois_hc_bs() -> None:
    """(10.7.2) : l'energie volumique d'un cycle est son aire, positive."""
    assert aire_du_cycle(CYCLE_RECTANGULAIRE) == pytest.approx(4 * 40.0 * 1.2)


def test_un_cycle_parcouru_a_l_envers_rendrait_de_l_energie() -> None:
    """« Essentiellement positive » : le sens du parcours porte le signe."""
    assert aire_du_cycle(CYCLE_RECTANGULAIRE[::-1]) == pytest.approx(-192.0)


def test_l_aire_d_un_cycle_elliptique_est_pi_hm_bm_sin_delta() -> None:
    """`H = H_m cos t`, `B = B_m cos(t - delta)` : l'induction retarde."""
    champ, induction, retard, pas = 300.0, 0.8, 0.4, 4000
    cycle = [
        (
            champ * math.cos(2 * math.pi * k / pas),
            induction * math.cos(2 * math.pi * k / pas - retard),
        )
        for k in range(pas)
    ]
    attendue = math.pi * champ * induction * math.sin(retard)
    assert aire_du_cycle(cycle) == pytest.approx(attendue, rel=1e-5)


def test_un_cycle_de_moins_de_trois_points_n_a_pas_d_aire() -> None:
    with pytest.raises(ValueError, match="au moins trois points"):
        aire_du_cycle([(0.0, 0.0), (1.0, 1.0)])


CYCLE_EN_LOSANGE = ((100.0, 1.0), (-50.0, 0.5), (-100.0, -1.0), (50.0, -0.5))
"""Quatre sommets : `B` croise zero sur les cotes, `H` aussi."""


def test_l_induction_remanente_se_lit_a_champ_nul() -> None:
    """Page 10-8 : « la valeur `B_r` de `B` pour `H = 0` ».

    Du sommet `(100 ; 1)` au sommet `(-50 ; 0,5)`, le champ s'annule aux
    deux tiers du cote : `B = 1 - 0,5 . 2/3 = 2/3`.
    """
    assert induction_remanente(CYCLE_EN_LOSANGE) == pytest.approx(2 / 3)


def test_le_champ_coercitif_se_lit_a_induction_nulle() -> None:
    """Page 10-8 : « la valeur `H_c` de `H` pour `B = 0` ».

    De `(-50 ; 0,5)` a `(-100 ; -1)`, l'induction s'annule au tiers du
    cote : `H = -50 - 50/3`, soit un champ coercitif de `200/3 A/m`.
    """
    assert champ_coercitif(CYCLE_EN_LOSANGE) == pytest.approx(200 / 3)


def test_un_cycle_qui_ne_coupe_pas_l_axe_n_a_pas_de_remanente() -> None:
    with pytest.raises(ValueError, match="ne coupe pas l'axe"):
        induction_remanente([(1.0, 1.0), (2.0, 1.0), (2.0, 2.0)])


def test_le_champ_demagnetisant_s_oppose_a_l_aimantation() -> None:
    """Page 10-29 : `N = H' / M`, et `H'` tend a s'opposer a `H_0`."""
    assert champ_demagnetisant(0.25, 800.0) == pytest.approx(-200.0)


def test_un_facteur_de_desaimantation_negatif_est_refuse() -> None:
    with pytest.raises(ValueError, match="facteur de desaimantation negatif"):
        champ_demagnetisant(-0.1, 800.0)


def test_le_champ_interieur_referme_la_boucle_sur_l_aimantation() -> None:
    """`H = H_0 - N M` avec `M = chi H` : les deux lectures s'accordent."""
    applique, facteur, relative = 1000.0, 0.25, 201.0
    interieur = champ_interieur(applique, facteur, relative)
    aimante = aimantation_induite(relative, interieur)
    assert interieur == pytest.approx(applique + champ_demagnetisant(facteur, aimante))
    assert interieur < applique


def test_le_champ_demagnetisant_approche_le_champ_applique_si_mu_est_grand() -> None:
    """Page 10-30, mot pour mot ; et negligeable pour un corps paramagnetique."""
    faible = part_du_champ_applique(0.25, CORPS_PARAMAGNETIQUES["aluminium"])
    moyenne = part_du_champ_applique(0.25, 201.0)
    forte = part_du_champ_applique(0.25, 20001.0)
    assert faible == pytest.approx(5.0e-6, rel=1e-3)
    assert moyenne == pytest.approx(50 / 51)
    assert moyenne < forte < 1


def test_le_champ_terrestre_plonge_vers_le_centre_de_la_terre() -> None:
    """Page 10-30 : 15 A/m a l'horizontale, 35 A/m a la verticale."""
    assert (COMPOSANTE_HORIZONTALE, COMPOSANTE_VERTICALE) == (15.0, 35.0)
    assert intensite(15.0, 35.0) == pytest.approx(math.sqrt(1450.0))
    assert math.degrees(inclinaison(15.0, 35.0)) == pytest.approx(66.80, abs=0.01)
    induction = induction_du_milieu(intensite(15.0, 35.0), 0.0)
    assert induction == pytest.approx(47.9e-6, rel=2e-3)


def test_un_champ_terrestre_nul_n_a_pas_d_inclinaison() -> None:
    with pytest.raises(ValueError, match="champ nul"):
        inclinaison(0.0, 0.0)


def test_a_la_temperature_de_curie_le_corps_n_est_plus_ferromagnetique() -> None:
    """Page 10-3 : « au-dessous de laquelle », au sens strict ; zero kelvin vaut."""
    assert not est_ferromagnetique(1050.0, TEMPERATURE_DE_CURIE_DU_FER)
    assert est_ferromagnetique(0.0, TEMPERATURE_DE_CURIE_DU_FER)


def test_le_travail_et_son_complement_remplissent_le_rectangle_h_b() -> None:
    """Hors du lineaire, `integrale de H dB` n'est plus `integrale de B dH`.

    Courbe d'essai `B = c racine(H)` : le travail (10.7.1) vaut
    `c H^(3/2) / 3`, son complement le double, et leur somme `H B`.
    """
    pente, final = sp.symbols("c H_1", positive=True)
    induction = pente * sp.sqrt(H)
    travail = travail_volumique(induction, H, 0, final)
    assert sp.simplify(travail - pente * final ** sp.Rational(3, 2) / 3) == 0
    complement = sp.integrate(induction, (H, 0, final))
    assert sp.simplify(travail + complement - final * induction.subs(H, final)) == 0


CYCLE_A_SOMMETS_SUR_L_AXE = ((100.0, 1.0), (0.0, 0.6), (-100.0, -1.0), (0.0, -0.6))
"""Deux sommets a champ nul : la remanente s'y lit sans interpoler."""


def test_un_sommet_pose_sur_l_axe_est_lu_tel_quel() -> None:
    """De `(0 ; 0,6)` a `(-100 ; -1)`, `B` s'annule a `0,6 / 1,6` du cote."""
    assert induction_remanente(CYCLE_A_SOMMETS_SUR_L_AXE) == pytest.approx(0.6)
    assert champ_coercitif(CYCLE_A_SOMMETS_SUR_L_AXE) == pytest.approx(37.5)


def test_un_corps_infiniment_allonge_ne_se_desaimante_pas() -> None:
    """Page 10-29 : le facteur est « d'autant plus petit que le corps est allonge »."""
    assert champ_demagnetisant(0.0, 800.0) == 0
    assert champ_interieur(1000.0, 0.0, 201.0) == pytest.approx(1000.0)


def test_un_champ_purement_vertical_est_incline_d_un_angle_droit() -> None:
    assert inclinaison(0.0, 35.0) == pytest.approx(math.pi / 2)
    assert inclinaison(15.0, 0.0) == 0
