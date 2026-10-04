"""Ce que les cours empruntent a la chimie : atome, quantite de matiere, ions.

Oracles : le syllabus d'electricite generale
(`Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`, pages
1-4 a 1-7) -- l'atome de cuivre des exemples 1.3.1 et 1.3.2, et l'acide
sulfurique de l'exemple 1.4.2. Le calcul y est refait a la main, avec
les valeurs que le syllabus emploie.
"""

from __future__ import annotations

import math

import pytest

from polytech.chemistry.domain.atomes import Atome
from polytech.chemistry.domain.ions import Dissociation, Ion
from polytech.chemistry.domain.matiere import (
    CONSTANTE_D_AVOGADRO,
    nombre_d_entites,
    quantite_de_matiere,
)

CUIVRE_63 = Atome("Cu", numero_atomique=29, nombre_de_masse=63)
CUIVRE_64 = Atome("Cu", numero_atomique=29, nombre_de_masse=64)


def test_un_atome_neutre_a_autant_d_electrons_que_de_protons() -> None:
    assert CUIVRE_63.protons == CUIVRE_63.electrons == 29


def test_les_neutrons_completent_le_nombre_de_masse() -> None:
    """Exemple 1.3.1 : `63 - 29 = 34` ou `64 - 29 = 35` neutrons."""
    assert CUIVRE_63.neutrons == 34
    assert CUIVRE_64.neutrons == 35


def test_deux_isotopes_ne_different_que_par_leurs_neutrons() -> None:
    assert CUIVRE_63.est_isotope_de(CUIVRE_64)
    assert not CUIVRE_63.est_isotope_de(CUIVRE_63)
    assert not CUIVRE_63.est_isotope_de(Atome("Zn", 30, 64))


def test_un_nombre_de_masse_inferieur_au_numero_atomique_est_refuse() -> None:
    with pytest.raises(ValueError, match="nombre de masse"):
        Atome("Cu", numero_atomique=29, nombre_de_masse=20)


def test_la_quantite_de_matiere_d_un_millimetre_cube_de_cuivre() -> None:
    """Exemple 1.3.2 : `rho V / M = 8960e-9 / 63,546e-3` mol."""
    quantite = quantite_de_matiere(masse=8960 * 1e-9, masse_molaire=63.546e-3)
    assert math.isclose(quantite, 8960e-9 / 63.546e-3)


def test_le_nombre_d_atomes_est_la_quantite_fois_avogadro() -> None:
    """Environ `8,5 . 10^19` atomes ; fois `Z = 29`, les `2,4 . 10^21` du syllabus."""
    atomes = nombre_d_entites(quantite_de_matiere(8960e-9, 63.546e-3))
    assert math.isclose(atomes, 8960e-9 / 63.546e-3 * CONSTANTE_D_AVOGADRO)
    assert 2.4e21 <= CUIVRE_63.protons * atomes < 2.5e21


def test_une_masse_molaire_nulle_est_refusee() -> None:
    with pytest.raises(ValueError, match="masse molaire"):
        quantite_de_matiere(masse=1.0, masse_molaire=0.0)


def test_un_cation_porte_des_charges_positives_un_anion_des_negatives() -> None:
    assert Ion("H", valence=1).est_un_cation
    assert Ion("SO4", valence=-2).est_un_anion


def test_un_ion_de_valence_nulle_est_refuse() -> None:
    with pytest.raises(ValueError, match="valence nulle"):
        Ion("H", valence=0)


def test_l_acide_sulfurique_se_scinde_en_deux_cations_et_un_anion() -> None:
    """Exemple 1.4.2 : `H2SO4 -> 2 H+ + SO4^2-`."""
    acide = Dissociation("H2SO4", ((2, Ion("H", 1)), (1, Ion("SO4", -2))))
    assert acide.charges_positives == 2
    assert acide.charges_negatives == 2


def test_une_dissociation_qui_cree_de_la_charge_est_refusee() -> None:
    """La molecule est neutre : ses ions doivent se compenser."""
    with pytest.raises(ValueError, match="electroneutralite"):
        Dissociation("H2SO4", ((1, Ion("H", 1)), (1, Ion("SO4", -2))))
