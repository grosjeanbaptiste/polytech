"""La presentation des laboratoires d'electronique : le potentiometre.

Source : `ElectroniqueInstrumentation/SLIDES/Slides_Presentation_des_labos.pdf`,
planche 19, lue sur sa couche de texte.

L'oracle : « on retrouve le calibre (100 kohm) entre A et C et des
resistances variables entre A et B (x ohm) et entre B et C (100 kohm - x) ».
"""

from __future__ import annotations

import pytest

from polytech.electronique.domain.amplification import gain_inverseur
from polytech.electronique.domain.potentiometre import (
    rapport_du_diviseur,
    resistances_du_potentiometre,
)

CALIBRE = 100e3
"""Planche 19 : le calibre du potentiometre du kit."""


def test_les_deux_resistances_somment_au_calibre() -> None:
    """Entre A et B, `x` ; entre B et C, le calibre moins `x`."""
    assert resistances_du_potentiometre(CALIBRE, 0.3) == pytest.approx((30e3, 70e3))
    for curseur in (0.0, 0.25, 1.0):
        gauche, droite = resistances_du_potentiometre(CALIBRE, curseur)
        assert gauche + droite == pytest.approx(CALIBRE)
    assert resistances_du_potentiometre(CALIBRE, 0.0) == (0.0, CALIBRE)


def test_le_curseur_fait_un_diviseur_de_tension() -> None:
    """Alimente entre A et C, le curseur B rend la fraction `x / calibre`."""
    assert rapport_du_diviseur(CALIBRE, 0.3) == pytest.approx(0.3)
    gauche, _ = resistances_du_potentiometre(CALIBRE, 0.3)
    assert rapport_du_diviseur(CALIBRE, 0.3) == pytest.approx(gauche / CALIBRE)


def test_un_potentiometre_en_contre_reaction_regle_le_gain_depuis_zero() -> None:
    """Deux pattes seulement : une resistance variable, de zero au calibre."""
    variable, _ = resistances_du_potentiometre(80e3, 0.5)
    assert gain_inverseur(2e3, variable) == pytest.approx(-20.0)
    nulle, _ = resistances_du_potentiometre(80e3, 0.0)
    assert nulle == 0.0


def test_les_gardes_du_potentiometre() -> None:
    """Un calibre nul ou un curseur hors de sa course sont refuses."""
    with pytest.raises(ValueError, match="calibre nul ou negatif"):
        resistances_du_potentiometre(0.0, 0.5)
    with pytest.raises(ValueError, match="curseur hors de"):
        resistances_du_potentiometre(CALIBRE, 1.5)
    with pytest.raises(ValueError, match="curseur hors de"):
        rapport_du_diviseur(CALIBRE, -0.1)
