"""Thermodynamique : le gaz parfait et la loi de Dalton, par la bibliotheque.

Phase 2, taxonomie d'abord : aucun code n'est ecrit pour ce cours. Ces tests
verifient que les symboles de `polytech.chemistry` auxquels
`cursus/notions/Thermodynamique.toml` rattache un titre portent ce que le
cours enonce.

Sources : `Thermodynamique/SYLLABUS/Syllabus_Bases_de_Thermodynamique_2024_2025.pdf`,
section 4.2, pages 51 et 52 ; `Syllabus_Thermodynamique_chimique_02_`
`Melange_de_gaz_parfaits.pdf`, pages 1 et 2 ; et la planche 12 du jeu
`Slides_Thermodynamique_chimique_02_..._Solutions_ideales.pdf`.
"""

from __future__ import annotations

import pytest

from polytech.chemistry.domain.gaz import (
    AIR_SEC,
    CONSTANTE_DES_GAZ,
    constante_specifique,
    masse_volumique_du_gaz_parfait,
    pression_partielle,
    quantite_de_gaz,
)

PRESSION, TEMPERATURE = 2.0e5, 298.15
"""Planche 12 : `p = 2 bar`, `T = 25 degres C`."""


def test_la_constante_universelle_est_celle_du_syllabus() -> None:
    """Page 52 : `R = 8.314 J/(mol K)`, et `r = R / M`, relation (4.5)."""
    assert pytest.approx(8.314, abs=5e-4) == CONSTANTE_DES_GAZ
    assert constante_specifique(AIR_SEC) * AIR_SEC == pytest.approx(CONSTANTE_DES_GAZ)


def test_le_volume_massique_est_r_t_sur_p() -> None:
    """Page 51, relation (4.1) : `v = r T / p`, donc `rho = p / (r T)`."""
    volume_massique = constante_specifique(AIR_SEC) * TEMPERATURE / PRESSION
    masse_volumique = masse_volumique_du_gaz_parfait(PRESSION, TEMPERATURE, AIR_SEC)
    assert masse_volumique * volume_massique == pytest.approx(1)


def test_le_volume_de_n_moles_est_n_r_t_sur_p() -> None:
    """Page 52, relation (4.4) : `V = n R T / p`."""
    moles = 8000.0
    volume = moles * CONSTANTE_DES_GAZ * TEMPERATURE / PRESSION
    assert quantite_de_gaz(PRESSION, volume, TEMPERATURE) == pytest.approx(moles)


def test_les_pressions_partielles_du_melange_de_la_planche_12() -> None:
    """Loi de Dalton : `p = p_1 + p_2`, avec `y = 0.625` et `0.375` a 2 bar."""
    du_dioxyde = pression_partielle(5000 / 8000, 2.0)
    de_l_oxygene = pression_partielle(3000 / 8000, 2.0)
    assert (du_dioxyde, de_l_oxygene) == pytest.approx((1.25, 0.75))
    assert du_dioxyde + de_l_oxygene == pytest.approx(2.0)
