"""L'hypothese de Boussinesq de la turbulence : `mu_t` ferme le tenseur de Reynolds.

Fiche, article « Decomposition de Reynolds » : le tenseur de Reynolds
`R_ij = -rho <u_i' u_j'>` est modelise comme une contrainte visqueuse,
`R_ij = mu_t (d_j u_i + d_i u_j) - (2/3) rho k delta_ij`. Oracles : un
cisaillement simple, et la trace `R_kk = -2 rho k` en incompressible.
"""

from __future__ import annotations

import pytest

from polytech.fluides.domain.viscosite_turbulente import tenseur_de_boussinesq

CISAILLEMENT = ((0.0, 2.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))  # d_y u_x = 2


def test_un_cisaillement_simple_donne_mu_t_fois_le_gradient() -> None:
    r = tenseur_de_boussinesq(0.5, CISAILLEMENT, 1000.0, 0.0)
    assert r[0][1] == r[1][0] == pytest.approx(1.0)


def test_sans_energie_cinetique_les_termes_diagonaux_sont_nuls() -> None:
    r = tenseur_de_boussinesq(0.5, CISAILLEMENT, 1000.0, 0.0)
    assert r[0][0] == r[1][1] == r[2][2] == 0.0


def test_la_trace_vaut_moins_deux_rho_k_en_incompressible() -> None:
    r = tenseur_de_boussinesq(0.5, CISAILLEMENT, 1000.0, 0.03)
    assert sum(r[i][i] for i in range(3)) == pytest.approx(-2 * 1000.0 * 0.03)


def test_le_tenseur_est_symetrique() -> None:
    gradient = ((1.0, 2.0, 3.0), (4.0, -5.0, 6.0), (7.0, 8.0, 4.0))
    r = tenseur_de_boussinesq(0.2, gradient, 1.2, 0.1)
    assert all(r[i][j] == r[j][i] for i in range(3) for j in range(3))


def test_une_viscosite_turbulente_nulle_ne_laisse_que_la_pression_turbulente() -> None:
    r = tenseur_de_boussinesq(0.0, CISAILLEMENT, 1000.0, 0.03)
    assert r[0][1] == 0.0
    assert r[0][0] == pytest.approx(-2 / 3 * 1000.0 * 0.03)


@pytest.mark.parametrize(
    ("mu_t", "rho", "k"), [(-1.0, 1.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, -0.1)]
)
def test_une_grandeur_non_physique_leve(mu_t: float, rho: float, k: float) -> None:
    with pytest.raises(ValueError, match=r"viscosite|masse volumique|energie"):
        tenseur_de_boussinesq(mu_t, CISAILLEMENT, rho, k)


def test_un_gradient_qui_n_est_pas_3_sur_3_leve() -> None:
    with pytest.raises(ValueError, match="gradient"):
        tenseur_de_boussinesq(0.5, ((0.0, 1.0), (0.0, 0.0)), 1000.0, 0.0)
