"""La limite exacte d'une fraction rationnelle a l'infini.

Trois cas, trois oracles ecrits a la main : une limite finie, une limite
nulle, une limite qui n'existe pas.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electronique.domain.limites import limite_a_l_infini

X, A, B = sp.symbols("x a b", positive=True)


def test_une_limite_finie_est_le_rapport_des_termes_dominants() -> None:
    assert sp.simplify(limite_a_l_infini((A * X + B) / (X + 1), X) - A) == 0


def test_une_limite_nulle_est_rendue_nulle() -> None:
    """`1 / x -> 0` : le numerateur l'emporte en `epsilon`."""
    assert limite_a_l_infini(1 / X, X) == 0


def test_une_limite_infinie_est_refusee() -> None:
    """`x -> oo` dans `x` lui-meme : la limite n'existe pas."""
    with pytest.raises(ValueError, match="limite infinie"):
        limite_a_l_infini(X, X)
