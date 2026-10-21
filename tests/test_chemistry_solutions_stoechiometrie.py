"""Solutions et stoechiometrie, d'apres « Introduction aux sciences de l'ingenieur ».

Source : `IntroductionSciencesIngenieur/SYLLABUS/`
`Syllabus_Introduction_aux_sciences_de_l_ingenieur_2025_2026.pdf`,
chapitres K et L (pages imprimees 97 a 103). Chaque oracle est un exemple
chiffre du syllabus ; chaque test le refait aussi par une conservation --
les moles de solute a la dilution, les atomes et les quantites a la reaction.
"""

from __future__ import annotations

import pytest

from polytech.chemistry.domain.matiere import quantite_de_matiere
from polytech.chemistry.domain.solutions import molarite, volume_de_solution_mere
from polytech.chemistry.domain.stoechiometrie import (
    produit_forme,
    quantites_restantes,
    reactif_limitant,
)


def test_molarite_de_la_soude_k_3_1() -> None:
    """P. 97 : 11,5 g de NaOH (40,0 g/mol) dans 1,50 l donnent 0,192 mol/l."""
    n = quantite_de_matiere(11.5, 40.0)
    assert molarite(n, 1.50) == pytest.approx(0.192, abs=5e-4)


def test_volume_de_solution_mere_k_2() -> None:
    """P. 98 : 500 ml a 1,00 mol/l depuis 17,4 mol/l demandent 28,7 ml."""
    v = volume_de_solution_mere(17.4, 1.00, 0.500)
    assert v == pytest.approx(0.0287, abs=5e-5)
    # Second chemin : les moles de solute sont les memes avant et apres.
    assert 17.4 * v == pytest.approx(1.00 * 0.500)


def test_sans_dilution_le_volume_est_le_meme() -> None:
    """La meme concentration : on preleve le volume vise, sans eau."""
    assert volume_de_solution_mere(2.0, 2.0, 0.5) == pytest.approx(0.5)


def test_une_dilution_ne_concentre_pas() -> None:
    with pytest.raises(ValueError, match="dilution"):
        volume_de_solution_mere(1.0, 2.0, 0.5)


def test_molarite_refuse_un_volume_nul() -> None:
    with pytest.raises(ValueError, match="volume"):
        molarite(1.0, 0.0)


LIOH = {"LiOH": 2, "CO2": 1}


def test_le_co2_absorbe_par_un_kilo_de_lioh_l_2_3() -> None:
    """P. 102 : 1,00 kg de LiOH (23,95 g/mol) absorbe 20,9 mol, soit 920 g de CO2."""
    n_lioh = quantite_de_matiere(1000.0, 23.95)
    n_co2 = produit_forme(n_lioh, LIOH["LiOH"], LIOH["CO2"])
    assert n_co2 == pytest.approx(20.9, abs=0.05)
    # Ecart au cours, d'arrondi : le syllabus arrondit d'abord a 20,9 mol, puis
    # 20,9 x 44,01 = 919,8 g, « 920 g ». Sans arrondi intermediaire, 918,8 g.
    assert round(n_co2, 1) * 44.01 == pytest.approx(920.0, abs=0.5)
    assert n_co2 * 44.01 == pytest.approx(918.8, abs=0.05)


HABER = {"N2": 1, "H2": 3}


def test_le_dihydrogene_limite_le_procede_haber() -> None:
    """P. 103 : 25,0 kg de N2 et 5,00 kg de H2 ; le H2 est limitant."""
    quantites = {
        "N2": quantite_de_matiere(25_000.0, 28.02),
        "H2": quantite_de_matiere(5_000.0, 2.016),
    }
    assert reactif_limitant(quantites, HABER) == "H2"
    n_nh3 = produit_forme(quantites["H2"], HABER["H2"], 2)
    assert n_nh3 == pytest.approx(1653, abs=1)
    assert n_nh3 * 17.031 / 1000 == pytest.approx(28.2, abs=0.05)


def test_le_limitant_est_epuise_les_autres_restent_positifs() -> None:
    """Second chemin : la reaction menee au bout vide le limitant, et lui seul."""
    quantites = {
        "N2": quantite_de_matiere(25_000.0, 28.02),
        "H2": quantite_de_matiere(5_000.0, 2.016),
    }
    restes = quantites_restantes(quantites, HABER)
    assert restes["H2"] == pytest.approx(0.0, abs=1e-9)
    assert restes["N2"] > 0
    # 892 mol de N2, dont 2480 / 3 reagissent.
    assert restes["N2"] == pytest.approx(quantites["N2"] - quantites["H2"] / 3)


def test_des_quantites_stoechiometriques_s_epuisent_ensemble() -> None:
    """P. 102 : en proportions stoechiometriques, tous les reactifs s'epuisent."""
    restes = quantites_restantes({"N2": 1.0, "H2": 3.0}, HABER)
    assert restes == pytest.approx({"N2": 0.0, "H2": 0.0})


def test_un_reactif_sans_coefficient_leve() -> None:
    with pytest.raises(ValueError, match="coefficient"):
        reactif_limitant({"N2": 1.0, "O2": 1.0}, HABER)
