"""Introduction aux sciences de l'ingenieur : chimie et matrices, par la bibliotheque.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Ces tests
verifient que les symboles auxquels
`cursus/notions/IntroductionSciencesIngenieur.toml` rattache un titre portent
ce que le syllabus enonce -- en recopiant ses exemples, page citee.

Source : `IntroductionSciencesIngenieur/SYLLABUS/`
`Syllabus_Introduction_aux_sciences_de_l_ingenieur_2025_2026.pdf`. Les pages
sont celles du PDF ; la page imprimee est entre parentheses.
"""

from __future__ import annotations

import pytest

from polytech.chemistry.domain.atomes import Atome
from polytech.chemistry.domain.ions import Dissociation, Ion
from polytech.chemistry.domain.matiere import (
    CONSTANTE_D_AVOGADRO,
    nombre_d_entites,
    quantite_de_matiere,
)
from polytech.chemistry.domain.reactions import composition, est_equilibree
from polytech.maths.domain.algebre import matrices

A = ((1.0, 2.0), (4.0, -1.0))
B = ((0.0, 1.0), (2.0, 3.0))
"""Page 46 (41), exemple 2 du produit de deux matrices."""


def test_l_atome_et_ses_isotopes_de_la_section_j_2() -> None:
    """Page 87 (82) : `A = Z + N` ; deux isotopes ont le meme `Z`."""
    carbone = Atome("C", 6, 12)
    assert carbone.protons + carbone.neutrons == 12
    assert carbone.electrons == carbone.protons
    assert Atome("C", 6, 14).est_isotope_de(carbone)
    assert not Atome("N", 7, 14).est_isotope_de(carbone)


def test_les_ions_et_les_electrolytes_des_sections_j_2_2_et_k_2_2() -> None:
    """Page 102 (97) : `HCl -> H+ + Cl-`, `NaOH -> Na+ + OH-`, sans creer de charge."""
    assert Ion("Na", 1).est_un_cation
    assert Ion("Cl", -1).est_un_anion
    acide = Dissociation("HCl", ((1, Ion("H", 1)), (1, Ion("Cl", -1))))
    assert acide.charges_positives == acide.charges_negatives == 1
    with pytest.raises(ValueError, match="electroneutralite violee pour NaOH"):
        Dissociation("NaOH", ((1, Ion("Na", 1)), (2, Ion("OH", -1))))


def test_la_mole_et_le_nombre_d_avogadro_de_la_section_j_7() -> None:
    """Pages 97 et 108 (92 et 103) : `N_A = 6,02 10^23` ; 25,0 kg de N2, 892 mol."""
    assert pytest.approx(6.02e23, rel=1e-3) == CONSTANTE_D_AVOGADRO
    assert nombre_d_entites(1) == CONSTANTE_D_AVOGADRO
    assert quantite_de_matiere(25.0, 28.02e-3) == pytest.approx(892, abs=0.5)
    assert quantite_de_matiere(5.00, 2.016e-3) == pytest.approx(2480, abs=0.5)


def test_la_composition_de_l_ethanol_de_la_section_j_8() -> None:
    """Page 98 (93) : `C2H5OH` contient 2 C, 6 H et 1 O ; 52,14 % de carbone."""
    atomes = composition("C2H5OH")
    assert atomes == {"C": 2, "H": 6, "O": 1}
    masses = {"C": 12.01, "H": 1.008, "O": 16.00}
    molaire = sum(masses[element] * nombre for element, nombre in atomes.items())
    assert molaire == pytest.approx(46.07, abs=0.005)
    assert 100 * atomes["C"] * masses["C"] / molaire == pytest.approx(52.14, abs=0.005)


def test_la_combustion_du_butane_s_equilibre_par_la_conservation_des_atomes() -> None:
    """Page 106 (101) : `2 C4H10 + 13 O2 -> 8 CO2 + 10 H2O`."""
    assert est_equilibree({"C4H10": 2, "O2": 13}, {"CO2": 8, "H2O": 10})
    assert not est_equilibree({"C4H10": 1, "O2": 6}, {"CO2": 4, "H2O": 5})
    assert est_equilibree({"N2": 1, "H2": 3}, {"NH3": 2})


def test_le_produit_de_deux_matrices_n_est_pas_commutatif() -> None:
    """Page 46 (41), exemple 2 : `AB = [[4, 7], [-2, 1]]`, `BA = [[4, -1], [14, 1]]`."""
    assert matrices.produit(A, B) == ((4, 7), (-2, 1))
    assert matrices.produit(B, A) == ((4, -1), (14, 1))
    assert matrices.produit(A, matrices.identite(2)) == A


def test_la_somme_le_produit_par_un_scalaire_et_la_transposee() -> None:
    """Pages 44 a 47 (39 a 42) : terme a terme ; la transposee de l'exemple 3."""
    assert matrices.addition(A, B) == ((1, 3), (6, 2))
    assert matrices.multiplication_par_un_scalaire(2, A) == ((2, 4), (8, -2))
    carre = ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0))
    assert matrices.transposee(carre) == ((1, 4, 7), (2, 5, 8), (3, 6, 9))
    assert matrices.transposee(matrices.transposee(carre)) == carre
