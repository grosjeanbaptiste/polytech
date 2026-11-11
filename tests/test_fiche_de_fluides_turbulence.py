"""Les articles de turbulence et de trainee de la fiche de fluides, refaits.

Decomposition de Reynolds, tenseur de Reynolds, contrainte et loi de
paroi, trainee de Stokes : chaque ligne de
`cursus/fiches/MecaniqueFluides.toml` pour ces notions est recalculee par
`polytech.fluides`, puis comparee par `polytech.maths.domain.comparaison`.
"""

from __future__ import annotations

import tomllib

import pytest
import sympy as sp

from polytech.fluides.domain.aerodynamique import Ecoulement, SurfacePortante
from polytech.fluides.domain.conduite import profil_de_poiseuille
from polytech.fluides.domain.grandeurs import MasseVolumique
from polytech.fluides.domain.paroi_turbulente import (
    contrainte_reduite_en_conduite,
    energie_cinetique_turbulente,
    tenseur_de_reynolds,
)
from polytech.fluides.domain.regime_de_reynolds import nombre_de_reynolds
from polytech.fluides.domain.trainee import coefficient_de_stokes
from polytech.fluides.domain.turbulence import (
    CONSTANTE_LOG,
    KARMAN,
    Decomposition,
    covariance,
    vitesse_adimensionnee,
)
from polytech.maths.domain.comparaison import egales, proches, suite_nulle
from tests.fiches_toml import FICHES

FICHE = FICHES / "MecaniqueFluides.toml"
UX = Decomposition((1.0, 1.4, 0.7, 1.3, 0.6))
UY = Decomposition((0.2, -0.3, 0.1, 0.4, -0.5))
UZ = Decomposition((0.0, 0.5, -0.2, 0.1, -0.4))
RHO = 1.2


def _produit(a: Decomposition, b: Decomposition) -> Decomposition:
    return Decomposition(
        tuple(x * y for x, y in zip(a.echantillons, b.echantillons, strict=True))
    )


def test_moyenne_de_fluctuation_nulle_et_moyenne_de_moyenne() -> None:
    assert suite_nulle([Decomposition(UX.fluctuations()).moyenne()], 1e-12)
    constante = Decomposition((UX.moyenne(),) * 5)
    assert proches(constante.moyenne(), UX.moyenne(), 1e-12)


def test_la_moyenne_d_un_produit_est_le_produit_des_moyennes_plus_la_covariance() -> (
    None
):
    attendue = UX.moyenne() * UY.moyenne() + covariance(UX, UY)
    assert proches(_produit(UX, UY).moyenne(), attendue, 1e-12)
    assert not proches(_produit(UX, UY).moyenne(), UX.moyenne() * UY.moyenne(), 1e-3)


def test_le_tenseur_de_reynolds_est_moins_rho_fois_la_covariance_des_fluctuations() -> (
    None
):
    composantes = (UX, UY, UZ)
    r = tenseur_de_reynolds(RHO, composantes)
    for i, a in enumerate(composantes):
        for j, b in enumerate(composantes):
            assert proches(float(r[i, j]), -RHO * covariance(a, b), 1e-12)


def test_l_energie_cinetique_de_turbulence_est_moins_la_trace_sur_deux_rho() -> None:
    r = tenseur_de_reynolds(RHO, (UX, UY, UZ))
    k = energie_cinetique_turbulente((UX, UY, UZ))
    assert proches(k, float(-r.trace()) / (2 * RHO), 1e-12)
    assert proches(k, (UX.variance() + UY.variance() + UZ.variance()) / 2, 1e-12)


@pytest.mark.parametrize("hauteur", [0.0, 0.1, 0.25, 0.5, 0.9, 1.0])
def test_la_contrainte_totale_est_lineaire_nulle_au_centre_maximale_a_la_paroi(
    hauteur: float,
) -> None:
    """Poiseuille : `tau = G r / 2`, et `y = R - r` se compte depuis la paroi."""
    r, rayon, mu, g = sp.symbols("r R mu G", positive=True)
    contrainte = -mu * sp.diff(profil_de_poiseuille(r, rayon, mu, g), r)
    reduite = contrainte / contrainte.subs(r, rayon)
    y = hauteur * rayon
    assert egales(reduite.subs(r, rayon - y), 1 - y / rayon)
    assert proches(
        contrainte_reduite_en_conduite(1.0 - hauteur, 1.0), 1 - hauteur, 1e-12
    )


def test_la_loi_de_paroi_a_les_constantes_et_les_seuils_de_la_fiche() -> None:
    assert (KARMAN, CONSTANTE_LOG) == (0.4, 5.0)
    assert all(proches(vitesse_adimensionnee(y), y, 1e-12) for y in (1.0, 4.9))
    for y in (30.0, 100.0, 1000.0):
        assert proches(vitesse_adimensionnee(y), float(sp.log(y)) / 0.4 + 5.0, 1e-12)
    with pytest.raises(ValueError, match="zone tampon"):
        vitesse_adimensionnee(10.0)


def test_la_figure_de_la_loi_logarithmique_prend_kappa_et_b_du_cours() -> None:
    articles = tomllib.loads(FICHE.read_text(encoding="utf-8"))["article"]
    figure = next(
        a for a in articles if a["notion"].startswith("ecoulement_turbulent")
    )["figure"]
    assert f"ln(x)/{KARMAN:g} + {CONSTANTE_LOG:g}" in figure


def test_la_trainee_de_stokes_est_24_sur_re_et_vaut_trois_pi_mu_d_v() -> None:
    mu, rho, d, v = sp.Rational(1, 10), 900, sp.Rational(1, 1000), sp.Rational(1, 50)
    reynolds = float(nombre_de_reynolds(float(v), float(d), float(mu / rho)))
    cd = coefficient_de_stokes(reynolds)
    assert reynolds < 1
    assert proches(cd, 24 / reynolds, 1e-12)
    surface = float(sp.pi * d**2 / 4)
    corps = SurfacePortante(Ecoulement(MasseVolumique(float(rho)), float(v)), surface)
    assert egales(corps.force(cd), 3 * sp.pi * mu * d * v)
    with pytest.raises(ValueError, match="Stokes"):
        coefficient_de_stokes(2.0)
