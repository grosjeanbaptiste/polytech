"""La convection naturelle de la fiche de thermique, refaite par `polytech/`."""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.maths.domain.comparaison import egales, proches, proches_relativement
from polytech.thermique.domain.convection import naturelle
from polytech.thermique.domain.convection.naturelle_externe import (
    conductance_de_la_lame,
    nusselt_de_la_paroi_verticale,
    nusselt_de_la_plaque_horizontale,
    nusselt_du_cylindre,
)

PR, RE = 0.71, 8.0e4


def test_le_grashof_et_le_beta_du_gaz_parfait() -> None:
    """`Gr = g beta dT L^3 / nu^2` ; gaz parfait : `beta = 1/T`."""
    t, p, r = sp.symbols("T p r", positive=True)
    beta = sp.simplify(-sp.diff(p / (r * t), t) / (p / (r * t)))
    assert egales(beta, 1 / t)
    gr = naturelle.grashof(1 / 300.0, 20.0, 0.5, 1.6e-5)
    assert proches_relativement(
        gr, 9.81 * (1 / 300.0) * 20.0 * 0.5**3 / 1.6e-5**2, 2e-3
    )


def test_la_convection_naturelle_externe_de_la_fiche() -> None:
    """Paroi `0,56 Ra^(1/4)` puis `0,12 Ra^(1/3)` ; cylindre `0,47` ; plaque `0,137`."""
    assert proches_relativement(
        nusselt_de_la_paroi_verticale(1e7), 0.56 * 1e7**0.25, 1e-12
    )
    assert proches_relativement(
        nusselt_de_la_paroi_verticale(1e10), 0.12 * 1e10 ** (1 / 3), 1e-12
    )
    assert proches_relativement(
        nusselt_du_cylindre(1e7 / PR, PR), 0.47 * 1e7**0.25, 1e-12
    )
    assert proches_relativement(
        nusselt_de_la_plaque_horizontale(1e8), 0.137 * 1e8 ** (1 / 3), 1e-12
    )


def test_dans_un_espace_ferme_nu_b_en_loi_puissance_et_conduction_sous_1400() -> None:
    """`K = Nu_b lam / b`, `Nu_b = 0,11 Ra^0,29` ; `Ra < 1400 : Nu_b = 1`."""
    ra, lam, b = 5.0e4, 0.026, 0.05
    assert proches_relativement(
        naturelle.nusselt_lame_de_fluide(ra), 0.11 * ra**0.29, 1e-12
    )
    assert proches_relativement(
        conductance_de_la_lame(ra, lam, b), 0.11 * ra**0.29 * lam / b, 1e-12
    )
    assert naturelle.nusselt_lame_de_fluide(1399.0) == 1.0
    with pytest.raises(ValueError, match="transition"):
        naturelle.nusselt_lame_de_fluide(3000.0)
    assert proches((0.30 - 2 * 0.10) / 2, 0.05, 1e-12)  # la marmite de mai 2023
