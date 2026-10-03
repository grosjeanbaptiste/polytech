"""La transformation triangle-etoile (theoreme de Kennelly).

Elle reduit un pont, que les seules associations serie et parallele ne
savent pas reduire. L'oracle n'est pas la formule : c'est ce qu'elle doit
conserver, la resistance vue entre deux sommets quand le troisieme est
en l'air, pour chacune des trois paires.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.circuits.etoile import triangle_vers_etoile

A, B, C = sp.symbols("R_ab R_bc R_ca", positive=True)


def _parallele(premiere: sp.Expr, seconde: sp.Expr) -> sp.Expr:
    return premiere * seconde / (premiere + seconde)


@pytest.mark.parametrize(
    ("cote", "chemin", "branches"),
    [
        (A, B + C, (0, 1)),
        (B, C + A, (1, 2)),
        (C, A + B, (2, 0)),
    ],
)
def test_l_etoile_conserve_la_resistance_entre_deux_sommets(
    cote: sp.Expr, chemin: sp.Expr, branches: tuple[int, int]
) -> None:
    etoile = triangle_vers_etoile(A, B, C)
    vue = etoile[branches[0]] + etoile[branches[1]]
    assert sp.simplify(vue - _parallele(cote, chemin)) == 0


def test_un_triangle_egal_donne_une_etoile_au_tiers() -> None:
    assert triangle_vers_etoile(3, 3, 3) == (1, 1, 1)


def test_un_triangle_a_cote_nul_est_refuse() -> None:
    with pytest.raises(ValueError, match="nulle"):
        triangle_vers_etoile(0, 3, 3)
