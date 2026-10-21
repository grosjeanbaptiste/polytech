"""L'ecoulement en conduite : Poiseuille, Darcy-Weisbach, Blasius.

Le chapitre V du cours (« Ecoulements en conduite ») manque au depot. Les
oracles sont donc ceux que le depot porte :

* l'examen de janvier 2025, `MecaniqueFluides/EXAMENS/Exam_2025_Janvier.pdf`,
  page 1, question 1.2 (Poiseuille, viscosite divisee par deux a debit
  constant) et page 5, question 5.2 (`lambda = 0,316 Re^(-0,25)`, la perte
  de charge du toboggan) ;
* les planches des equations fondamentales,
  `MecaniqueFluides/SLIDES/Slides_03_Equations_fondamentales.pdf`
  (Bernoulli avec un terme de pertes `Delta p_tot`, toujours positif) ;
* Navier-Stokes, que le cours resout pour Couette : le profil de Poiseuille
  est refait ici par sympy, sans la formule, comme second chemin.
"""

from __future__ import annotations

import math

import pytest
import sympy as sp

from polytech.fluides.domain.conduite import (
    coefficient_de_blasius,
    coefficient_laminaire,
    debit_de_poiseuille,
    perte_de_charge_de_poiseuille,
    perte_de_charge_reguliere,
    profil_de_poiseuille,
)


def test_le_profil_est_la_solution_de_navier_stokes_en_conduite() -> None:
    """`(mu / r) (r u')' = -G`, `u(R) = 0`, `u` borne en 0."""
    r, rayon, mu, g = sp.symbols("r R mu G", positive=True)
    u = profil_de_poiseuille(r, rayon, mu, g)
    assert sp.simplify(mu / r * sp.diff(r * sp.diff(u, r), r) + g) == 0
    assert u.subs(r, rayon) == 0


def test_le_debit_est_l_integrale_du_profil() -> None:
    r, rayon, mu, g = sp.symbols("r R mu G", positive=True)
    integrale = sp.integrate(
        2 * sp.pi * r * profil_de_poiseuille(r, rayon, mu, g), (r, 0, rayon)
    )
    assert sp.simplify(integrale - debit_de_poiseuille(rayon, mu, g)) == 0


def test_a_debit_constant_la_perte_suit_la_viscosite() -> None:
    """Janvier 2025, 1.2 : viscosite divisee par deux, perte divisee par deux."""
    pleine = perte_de_charge_de_poiseuille(1e-3, 0.05, 2.0, 0.01)
    moitie = perte_de_charge_de_poiseuille(1e-3, 0.05, 2.0, 0.005)
    assert math.isclose(moitie / pleine, 0.5)


def test_la_perte_de_poiseuille_est_celle_de_darcy_avec_64_sur_re() -> None:
    """Deux chemins : Hagen-Poiseuille et Darcy-Weisbach a `lambda = 64 / Re`."""
    debit, rayon, longueur, viscosite, rho = 1e-5, 0.01, 3.0, 1e-3, 1000.0
    vitesse = debit / (math.pi * rayon**2)
    reynolds = rho * vitesse * 2 * rayon / viscosite
    par_darcy = perte_de_charge_reguliere(
        coefficient_laminaire(reynolds), longueur, 2 * rayon, vitesse, rho
    )
    assert math.isclose(
        perte_de_charge_de_poiseuille(debit, rayon, longueur, viscosite), par_darcy
    )


def test_blasius_sur_le_toboggan_de_janvier_2025() -> None:
    """Question 5.2 : `Re = 9,0 x 10^4`, `lambda = 0,018`."""
    assert f"{coefficient_de_blasius(9.0e4):.2g}" == "0.018"


def test_blasius_est_refuse_hors_de_son_domaine_turbulent_lisse() -> None:
    with pytest.raises(ValueError, match="Blasius ne vaut qu'en turbulent"):
        coefficient_de_blasius(1500.0)


def test_le_coefficient_laminaire_est_refuse_en_turbulent() -> None:
    with pytest.raises(ValueError, match="64 / Re ne vaut qu'en laminaire"):
        coefficient_laminaire(9.0e4)


def test_une_perte_de_charge_est_toujours_positive() -> None:
    """Planches des equations fondamentales : le frottement est irreversible."""
    assert perte_de_charge_reguliere(0.018, 10.0, 0.15, 0.6, 1000.0) > 0
    with pytest.raises(ValueError, match="longueur ou diametre"):
        perte_de_charge_reguliere(0.018, -10.0, 0.15, 0.6, 1000.0)


def test_un_diametre_nul_est_refuse() -> None:
    with pytest.raises(ValueError, match="longueur ou diametre"):
        perte_de_charge_reguliere(0.018, 10.0, 0.0, 0.6, 1000.0)
