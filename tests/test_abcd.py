"""Les quadripoles en parametres ABCD : briques, cascade, gain charge.

Oracles independants de la construction : la cellule en T doit egaler le
produit serie x derivation x serie ; le gain d'une impedance serie
chargee doit redonner le diviseur de tension ; et une derivation seule,
chargee, ne doit rien diviser du tout.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electronique.domain.abcd import (
    cascade,
    cellule_en_t,
    derivation,
    gain_en_tension,
    serie,
)

Z, Z1, Z2, Z3, ZL = sp.symbols("Z Z_1 Z_2 Z_3 Z_L", positive=True)


def test_la_cellule_en_t_est_le_produit_de_ses_trois_briques() -> None:
    produit = cascade(serie(Z1), derivation(Z3), serie(Z2))
    assert sp.simplify(cellule_en_t(Z1, Z2, Z3) - produit) == sp.zeros(2, 2)


def test_la_cellule_en_t_a_les_parametres_de_l_enonce() -> None:
    m = cellule_en_t(Z1, Z2, Z3)
    assert sp.simplify(m[0, 0] - (1 + Z1 / Z3)) == 0
    assert sp.simplify(m[0, 1] - (Z1 + Z2 + Z1 * Z2 / Z3)) == 0
    assert sp.simplify(m[1, 0] - 1 / Z3) == 0
    assert sp.simplify(m[1, 1] - (1 + Z2 / Z3)) == 0


def test_une_impedance_serie_chargee_est_un_diviseur() -> None:
    assert sp.simplify(gain_en_tension(serie(Z), ZL) - ZL / (Z + ZL)) == 0


def test_une_derivation_chargee_ne_divise_rien() -> None:
    """Une source ideale impose sa tension a la derivation comme a la charge."""
    assert sp.simplify(gain_en_tension(derivation(Z), ZL) - 1) == 0


def test_une_cascade_vide_est_refusee() -> None:
    with pytest.raises(ValueError, match="aucun quadripole"):
        cascade()
