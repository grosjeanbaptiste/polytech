"""Les aires d'echange direct en deux dimensions, par les cordes croisees.

La methode de Hottel : entre deux surfaces planes d'une geometrie longue,
l'aire d'echange par metre de longueur vaut la demi-difference entre la
somme des cordes croisees et celle des cordes non croisees. Une corde est
un fil **tendu** : un obstacle l'oblige a s'enrouler sur ses bords.

Les oracles sont deux resultats que le cours tabule autrement :

- deux bandes paralleles de largeur `w` en vis-a-vis a la distance `h` :
  `racine(w^2 + h^2) - h` ;
- deux bandes perpendiculaires de largeurs `a` et `b` partageant une
  arete : `(a + b - racine(a^2 + b^2)) / 2`.
"""

from __future__ import annotations

import math

import pytest

from polytech.thermique.domain.rayonnement.cordes_croisees import (
    aire_par_les_cordes,
    corde_tendue,
)


def test_deux_bandes_paralleles_en_vis_a_vis() -> None:
    w, h = 1.2, 0.3
    croisees = (corde_tendue((0, h), (w, 0)), corde_tendue((w, h), (0, 0)))
    non_croisees = (corde_tendue((0, h), (0, 0)), corde_tendue((w, h), (w, 0)))
    assert aire_par_les_cordes(croisees, non_croisees) == pytest.approx(
        math.hypot(w, h) - h
    )


def test_deux_bandes_perpendiculaires_a_arete_commune() -> None:
    a, b = 0.6, 1.6
    coin, bout_horizontal, bout_vertical = (0.0, 0.0), (b, 0.0), (0.0, a)
    croisees = (corde_tendue(coin, bout_vertical), corde_tendue(bout_horizontal, coin))
    non_croisees = (
        corde_tendue(coin, coin),
        corde_tendue(bout_horizontal, bout_vertical),
    )
    assert aire_par_les_cordes(croisees, non_croisees) == pytest.approx(
        (a + b - math.hypot(a, b)) / 2
    )


def test_une_corde_tendue_s_enroule_sur_les_points_de_passage() -> None:
    """Le fil passe par le bord de l'obstacle : deux segments, pas un."""
    assert corde_tendue((0, 0), (3, 4), (6, 0)) == pytest.approx(10.0)


def test_une_aire_negative_trahit_des_cordes_echangees() -> None:
    with pytest.raises(ValueError, match="croisees"):
        aire_par_les_cordes((1.0, 1.0), (2.0, 2.0))
