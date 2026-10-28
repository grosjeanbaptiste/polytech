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


def test_la_garde_turbulente_suit_le_seuil_du_formulaire() -> None:
    """Changement assume : Colburn, Sieder-Tate et la transition disent `Re > 2400`."""
    assert (
        tube_turbulent.REYNOLDS_GARDE_TURBULENT
        == forcee.REYNOLDS_CRITIQUE_TUBE_FORMULAIRE_2019
    )


@pytest.mark.parametrize("reynolds", [2320.0, 2350.0, 2400.0])
def test_la_bande_2320_a_2400_est_refusee_comme_la_source(reynolds: float) -> None:
    with pytest.raises(ValueError, match="non turbulent"):
        tube_turbulent.nusselt_de_colburn(reynolds, 0.7)
    with pytest.raises(ValueError, match="non turbulent"):
        tube_turbulent.nusselt_turbulent_de_sieder_et_tate(reynolds, 0.7)
    with pytest.raises(ValueError, match="non turbulent"):
        tube_turbulent.nusselt_de_transition(reynolds, 0.7, elancement=0.1)


def test_juste_au_dessus_de_2400_colburn_calcule() -> None:
    assert tube_turbulent.nusselt_de_colburn(2400.5, 0.7) > 0
