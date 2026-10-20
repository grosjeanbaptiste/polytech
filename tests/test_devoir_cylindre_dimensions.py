"""Devoir cote de fluides : l'analyse dimensionnelle par les variables repetees.

`p - p_inf = f(rho, U, D, mu)` : cinq parametres, trois dimensions
primaires, donc deux groupes (Vaschy-Buckingham, eq. 7-14 du syllabus du
cours). Avec `rho, U, D` repetes, ils sont `C_p / 2` et `1 / Re`.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.application import devoir_cylindre_dimensions as dim


def test_cinq_parametres_et_trois_dimensions_donnent_deux_groupes() -> None:
    assert dim.nombre_de_groupes_attendu() == 2
    assert len(dim.groupes()) == 2


def test_le_groupe_de_la_pression_est_la_moitie_du_coefficient_de_pression() -> None:
    pression, _ = dim.groupes()
    assert pression == {"dp": 1, "rho": -1, "U": -2}


def test_le_groupe_de_la_viscosite_est_l_inverse_du_reynolds() -> None:
    _, viscosite = dim.groupes()
    assert viscosite == {"mu": 1, "rho": -1, "U": -1, "D": -1}


def test_les_groupes_sont_sans_dimension() -> None:
    for groupe in dim.groupes():
        assert dim.dimensions(groupe) == {}


def test_la_celerite_du_son_ajoute_le_mach() -> None:
    """Si la compressibilite compte, `c` entre et le troisieme groupe est `Ma`."""
    *_, celerite = dim.groupes(compressible=True)
    assert celerite == {"c": 1, "U": -1}
    assert sp.Integer(len(dim.groupes(compressible=True))) == 3
