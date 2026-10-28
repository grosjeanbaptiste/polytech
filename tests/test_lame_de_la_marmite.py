"""La lame d'air de la marmite norvegienne (mai 2023) se deduit de la geometrie.

Donnees (corpus `Exam_2022_2025_Exercices_cotes_resolus.md`) : caisse a base
carree de 30 cm de cote interieur, casserole cylindrique de 10 cm de rayon,
posee au milieu.
"""

from __future__ import annotations

import pytest

from polytech.maths.domain.comparaison import proches
from polytech.thermique.domain.lame_de_marmite import epaisseur_de_la_lame


def test_la_lame_est_l_intervalle_entre_la_casserole_et_la_caisse() -> None:
    """`(0,30 - 2 x 0,10) / 2 = 0,05 m`."""
    assert proches(epaisseur_de_la_lame(0.30, 0.10), 0.05, 1e-12)


def test_une_casserole_qui_touche_la_paroi_n_a_pas_de_lame() -> None:
    """Rayon egal a la demi-largeur : refuse, la lame serait nulle."""
    with pytest.raises(ValueError, match="aucune lame"):
        epaisseur_de_la_lame(0.30, 0.15)


def test_une_casserole_plus_large_que_la_caisse_est_refusee() -> None:
    """Elle ne rentre pas : refuse."""
    with pytest.raises(ValueError, match="aucune lame"):
        epaisseur_de_la_lame(0.30, 0.20)
