"""Deux seuils laminaire/turbulent en tube, chacun nomme par sa source.

Le tome 2 du syllabus de thermique ecrit 2320 a la section 1.3.2 (la
transition qui fixe la longueur d'etablissement) et 2400 au chapitre 3
(convection forcee) ; le formulaire de 2019, les examens et la fiche
disent 2400. Ce sont deux valeurs de source, pas un oubli : chacune a sa
constante, et ce test dit laquelle sert ou et de combien elles different.
"""

from __future__ import annotations

import pytest

from polytech.thermique.domain.convection import couches_limites, forcee, tube_turbulent


def test_la_transition_hydrodynamique_du_tome_2_vaut_2320() -> None:
    assert couches_limites.REYNOLDS_TRANSITION_EN_TUBE_TOME_2_SECTION_1_3_2 == 2320.0


def test_le_seuil_de_convection_forcee_du_formulaire_vaut_2400() -> None:
    assert forcee.REYNOLDS_CRITIQUE_TUBE_FORMULAIRE_2019 == 2400


def test_l_alias_historique_est_la_constante_nommee() -> None:
    assert (
        couches_limites.REYNOLDS_CRITIQUE_EN_TUBE
        == couches_limites.REYNOLDS_TRANSITION_EN_TUBE_TOME_2_SECTION_1_3_2
    )
    assert (
        forcee.REYNOLDS_CRITIQUE_TUBE == forcee.REYNOLDS_CRITIQUE_TUBE_FORMULAIRE_2019
    )


def test_l_ecart_entre_les_deux_sources_est_de_80() -> None:
    ecart = (
        forcee.REYNOLDS_CRITIQUE_TUBE_FORMULAIRE_2019
        - couches_limites.REYNOLDS_TRANSITION_EN_TUBE_TOME_2_SECTION_1_3_2
    )
    assert ecart == 80


def test_la_longueur_d_etablissement_suit_le_seuil_de_la_section_1_3_2() -> None:
    couches_limites.longueur_d_etablissement(0.02, 2320.0)
    with pytest.raises(ValueError, match="turbulent"):
        couches_limites.longueur_d_etablissement(0.02, 2321.0)


def test_colburn_suit_aujourd_hui_le_seuil_de_la_section_1_3_2() -> None:
    """Etat constate, pas decision : le formulaire dit `Re > 2400`."""
    with pytest.raises(ValueError, match="non turbulent"):
        tube_turbulent.nusselt_de_colburn(2320.0, 0.7)
    assert tube_turbulent.nusselt_de_colburn(2350.0, 0.7) > 0
