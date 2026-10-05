"""Syllabus d'electricite generale, chapitre 12 : les aimants permanents.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 12-1 a 12-5, lues sur les pages rendues.

L'oracle est un tore de dimensions choisies, dont l'etat est calcule deux
fois : par la droite (12.1.4) coupee avec la courbe de desaimantation,
puis par le bilan (12.2.1) qui ne passe par aucune reluctance.
"""

from __future__ import annotations

import pytest

from polytech.electromagnetisme.domain.constantes import PERMEABILITE_DU_VIDE
from polytech.electromagnetisme.domain.magnetostatique.aimant_torique import (
    champ_dans_l_aimant,
    pente_de_la_droite_de_charge,
    point_de_fonctionnement,
)
from polytech.electromagnetisme.domain.magnetostatique.aimants import (
    induction_sur_la_droite_de_desaimantation,
    point_de_fonctionnement_optimal,
    produit_d_energie,
    produit_maximal,
    volume_minimal_de_l_aimant,
)
from polytech.electromagnetisme.domain.magnetostatique.circuits_magnetiques import (
    reluctance,
)
from polytech.electromagnetisme.domain.magnetostatique.materiaux_pour_aimants import (
    COERCITIF_IMPRIME_DES_FERRITES,
    MATERIAUX,
    MateriauPourAimant,
    borne_du_produit_d_energie,
)

REMANENTE, COERCITIF = 1.0, 5.0e4
"""Un Alnico du tableau 12.3.1 : un tesla, cinquante kiloamperes par metre."""

LONGUEUR, SECTION = 0.2, 4.0e-4
ENTREFER, SECTION_ENTREFER = 2.0e-3, 4.0e-4
RELUCTANCE_ENTREFER = reluctance(ENTREFER, SECTION_ENTREFER)
PENTE = pente_de_la_droite_de_charge(LONGUEUR, SECTION, RELUCTANCE_ENTREFER)


def test_la_droite_d_a_pour_pente_moins_l_sur_re_s() -> None:
    """(12.1.4), soit `-mu_0 (l / l_e) (S_e / S)` pour un entrefer uniforme."""
    assert pytest.approx(-PERMEABILITE_DU_VIDE * 100) == PENTE


def test_une_geometrie_sans_longueur_n_a_pas_de_droite() -> None:
    with pytest.raises(ValueError, match="geometrie invalide"):
        pente_de_la_droite_de_charge(0.0, SECTION, RELUCTANCE_ENTREFER)
    with pytest.raises(ValueError, match="geometrie invalide"):
        pente_de_la_droite_de_charge(LONGUEUR, 0.0, RELUCTANCE_ENTREFER)
    with pytest.raises(ValueError, match="geometrie invalide"):
        pente_de_la_droite_de_charge(LONGUEUR, SECTION, 0.0)


def test_le_champ_dans_l_aimant_est_demagnetisant() -> None:
    """(12.1.3) : `H = -R_e S B / l`, de signe oppose a l'induction."""
    champ = champ_dans_l_aimant(0.5, LONGUEUR, SECTION, RELUCTANCE_ENTREFER)
    assert champ == pytest.approx(-0.5 / (PERMEABILITE_DU_VIDE * 100))
    assert champ * PENTE == pytest.approx(0.5)


def test_le_point_3_est_sur_la_droite_et_sur_la_courbe() -> None:
    """Figure 12.1.2 : l'intersection de `d` et de la courbe de desaimantation."""
    champ, induction = point_de_fonctionnement(REMANENTE, COERCITIF, PENTE)
    assert champ < 0 < induction < REMANENTE
    assert induction == pytest.approx(PENTE * champ)
    assert induction == pytest.approx(
        induction_sur_la_droite_de_desaimantation(champ, REMANENTE, COERCITIF)
    )


def test_une_droite_montante_n_est_pas_une_droite_de_charge() -> None:
    with pytest.raises(ValueError, match="pente positive ou nulle"):
        point_de_fonctionnement(REMANENTE, COERCITIF, 0.0)


def test_le_bilan_12_2_1_se_referme_sur_le_tore() -> None:
    """`B H V + B_e H_e V_e = 0`, sans passer par la reluctance.

    `Phi = B S = B_e S_e` donne `B_e`, et l'entrefer est de l'air :
    `H_e = B_e / mu_0`.
    """
    champ, induction = point_de_fonctionnement(REMANENTE, COERCITIF, PENTE)
    entrefer = induction * SECTION / SECTION_ENTREFER
    champ_entrefer = entrefer / PERMEABILITE_DU_VIDE
    volume, volume_entrefer = LONGUEUR * SECTION, ENTREFER * SECTION_ENTREFER
    bilan = induction * champ * volume + entrefer * champ_entrefer * volume_entrefer
    assert bilan == pytest.approx(0.0, abs=1e-12)
    # (12.1.1) : le theoreme d'Ampere, aucun courant n'etant enlace.
    assert champ * LONGUEUR + champ_entrefer * ENTREFER == pytest.approx(0.0, abs=1e-9)


def test_le_volume_12_2_2_est_celui_du_tore() -> None:
    """`V = -(1 / mu_0) B_e^2 V_e / (B H)` rend le volume dont on est parti."""
    champ, induction = point_de_fonctionnement(REMANENTE, COERCITIF, PENTE)
    entrefer = induction * SECTION / SECTION_ENTREFER
    volume = volume_minimal_de_l_aimant(
        entrefer, ENTREFER * SECTION_ENTREFER, produit_d_energie(induction, champ)
    )
    assert volume == pytest.approx(LONGUEUR * SECTION)


def test_des_pieces_polaires_relevent_l_induction() -> None:
    """Page 12-3 : `R'_e` plus petite, donc `d'` a droite de `d`."""
    avec = pente_de_la_droite_de_charge(LONGUEUR, SECTION, RELUCTANCE_ENTREFER / 2)
    _, sans_pieces = point_de_fonctionnement(REMANENTE, COERCITIF, PENTE)
    _, avec_pieces = point_de_fonctionnement(REMANENTE, COERCITIF, avec)
    assert avec_pieces > sans_pieces


def test_l_aimant_optimal_a_la_droite_de_pente_br_sur_hc() -> None:
    """12.2 : le point 3 au sommet du plus grand rectangle inscrit."""
    point = point_de_fonctionnement(REMANENTE, COERCITIF, -REMANENTE / COERCITIF)
    assert point == pytest.approx(point_de_fonctionnement_optimal(REMANENTE, COERCITIF))
    assert produit_d_energie(point[1], point[0]) == pytest.approx(
        produit_maximal(REMANENTE, COERCITIF)
    )
    autre = point_de_fonctionnement(REMANENTE, COERCITIF, PENTE)
    assert produit_d_energie(autre[1], autre[0]) < produit_maximal(REMANENTE, COERCITIF)


def test_le_produit_d_energie_ne_depasse_pas_br_hc() -> None:
    """Le rectangle est inscrit dans la courbe : ses cotes sont sous `H_c` et `B_r`."""
    assert borne_du_produit_d_energie(REMANENTE, COERCITIF) == pytest.approx(5.0e4)
    assert produit_maximal(REMANENTE, COERCITIF) <= borne_du_produit_d_energie(
        REMANENTE, COERCITIF
    )


def test_une_specification_negative_n_a_pas_de_borne() -> None:
    with pytest.raises(ValueError, match="specification negative"):
        borne_du_produit_d_energie(-1.0, COERCITIF)
    with pytest.raises(ValueError, match="specification negative"):
        borne_du_produit_d_energie(REMANENTE, -1.0)


def test_le_tableau_12_3_1_et_les_aimants_au_samarium() -> None:
    """Page 12-5, en unites SI : J/m^3, teslas, A/m."""
    assert set(MATERIAUX) == {"Alnico", "ferrites", "samarium-cobalt"}
    assert MATERIAUX["samarium-cobalt"].produit == (150e3, 250e3)
    assert MATERIAUX["samarium-cobalt"].coercitif[0] == 1000e3
    assert MATERIAUX["Alnico"].remanente == (1.0, 1.0)
    assert MATERIAUX["ferrites"].produit == (10e3, 20e3)
    for materiau in MATERIAUX.values():
        assert materiau.est_coherent()


def test_le_champ_coercitif_imprime_des_ferrites_est_impossible() -> None:
    """Le tableau imprime « 1 ou 2 centiemes » de kA/m : dix a vingt A/m.

    Avec moins d'un tesla, le produit d'energie plafonnerait a vingt
    joules par metre cube, contre les dix mille de la meme ligne. La note
    du tableau (« jusque quelques 300 a 500 kA/m ») et le paragraphe 10.10
    (« champ coercitif tres eleve ») disent des centaines.
    """
    assert COERCITIF_IMPRIME_DES_FERRITES == (10.0, 20.0)
    ferrites = MATERIAUX["ferrites"]
    plafond = borne_du_produit_d_energie(1.0, COERCITIF_IMPRIME_DES_FERRITES[1])
    assert plafond < ferrites.produit[0] / 100
    assert ferrites.coercitif == (100e3, 200e3)


def test_la_coherence_se_juge_sur_les_bornes_hautes() -> None:
    """Le plus petit `(HB)max` contre le plus grand `B_r H_c`, et zero vaut."""
    assert MateriauPourAimant((50e3, 60e3), (0.5, 1.0), (40e3, 80e3)).est_coherent()
    imprime = MateriauPourAimant(
        (10e3, 20e3), (0.1, 1.0), COERCITIF_IMPRIME_DES_FERRITES
    )
    assert not imprime.est_coherent()
    assert borne_du_produit_d_energie(0.0, COERCITIF) == 0
