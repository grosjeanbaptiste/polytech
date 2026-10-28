"""La convection de la fiche de thermique, refaite par `polytech/`."""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.maths.domain.comparaison import egales, proches_relativement
from polytech.thermique.domain.convection import forcee, similitude
from polytech.thermique.domain.convection.couches_limites import epaisseur_laminaire
from polytech.thermique.domain.convection.faisceau_de_tubes import vitesse_maximale
from polytech.thermique.domain.convection.plaque_plane import (
    rapport_des_couches_limites,
)
from polytech.thermique.domain.convection.tube_turbulent import (
    diametre_hydraulique_de_la_section,
    nusselt_de_colburn,
)
from polytech.thermique.domain.transitoire import CorpsTransitoire

PR, RE = 0.71, 8.0e4


def test_les_nombres_sans_dimension_de_la_fiche() -> None:
    """`Nu = K L/lam_f`, `Bi = K L/lam_s`, `Re = u L/nu`, `Pr = nu/a = mu c/lam`."""
    k, longueur, lam_f, lam_s = 12.0, 0.5, 0.026, 40.0
    assert proches_relativement(
        similitude.nusselt(k, longueur, lam_f), k * longueur / lam_f, 1e-12
    )
    corps = CorpsTransitoire(k, longueur, lam_s, 3.6e6)
    assert proches_relativement(corps.biot(), k * longueur / lam_s, 1e-12)
    assert proches_relativement(
        forcee.reynolds(3.0, longueur, 1.5e-5), 3.0 * longueur / 1.5e-5, 1e-12
    )
    nu, a, mu, c, lam = 1.5e-5, 2.1e-5, 1.8e-5, 1005.0, 0.026
    assert proches_relativement(forcee.prandtl(nu, a), nu / a, 1e-12)
    assert proches_relativement(
        similitude.prandtl_des_proprietes(mu, c, lam), mu * c / lam, 1e-12
    )


def test_les_couches_limites_de_la_fiche() -> None:
    """`delta_t / delta = Pr^(-1/3)` ; `delta / x = 4,64 Re_x^(-1/2)` (laminaire)."""
    assert proches_relativement(rapport_des_couches_limites(PR), PR ** (-1 / 3), 1e-12)
    assert proches_relativement(
        epaisseur_laminaire(2.0, 1.0e5) / 2.0, 4.64 * 1.0e5**-0.5, 1e-12
    )
    assert rapport_des_couches_limites(1.0) == 1.0


def test_la_plaque_le_nusselt_moyen_est_le_double_du_local() -> None:
    """`0,664 = 2 x 0,332` : `K(x) ~ x^(-1/2)` a pour moyenne le double de `K(L)`."""
    x, longueur = sp.symbols("x L", positive=True)
    local = 0.332 * x ** sp.Rational(-1, 2)
    moyen = sp.integrate(local, (x, 0, longueur)) / longueur
    assert egales(moyen, 2 * local.subs(x, longueur))
    assert proches_relativement(
        forcee.nusselt_plaque(RE, PR), 0.664 * RE**0.5 * PR ** (1 / 3), 1e-12
    )
    assert proches_relativement(
        forcee.nusselt_plaque(6.0e5, PR), 0.037 * 6.0e5**0.8 * PR ** (1 / 3), 1e-12
    )


def test_dans_un_tube_colburn_et_le_diametre_hydraulique() -> None:
    """`Re > 2400 : Nu = 0,0234 Re^0,8 Pr^(1/3)` ; `d_h = 4 A / P`."""
    assert forcee.REYNOLDS_CRITIQUE_TUBE == 2400
    assert proches_relativement(
        nusselt_de_colburn(RE, PR), 0.0234 * RE**0.8 * PR ** (1 / 3), 1e-12
    )
    cote_a, cote_b = 0.01, 0.04
    d_h = diametre_hydraulique_de_la_section(cote_a * cote_b, 2 * (cote_a + cote_b))
    assert proches_relativement(
        d_h, 4 * cote_a * cote_b / (2 * (cote_a + cote_b)), 1e-12
    )
    cercle = diametre_hydraulique_de_la_section(math.pi * 0.02**2 / 4, math.pi * 0.02)
    assert proches_relativement(cercle, 0.02, 1e-12)


def test_autour_d_un_tube_loi_de_puissance_par_plages_et_vitesse_maximale() -> None:
    """`Nu_d = C Re^n Pr^(1/3)` sans extrapolation ; `u_max = u_0 p_T/(p_T - d)`."""
    assert proches_relativement(
        forcee.nusselt_cylindre(1000.0, PR),
        0.683 * 1000.0**0.466 * PR ** (1 / 3),
        1e-12,
    )
    with pytest.raises(ValueError, match="hors des plages"):
        forcee.nusselt_cylindre(1.0e6, PR)
    assert proches_relativement(
        vitesse_maximale(2.0, pas=0.05, diametre=0.02), 2.0 * 0.05 / 0.03, 1e-12
    )
