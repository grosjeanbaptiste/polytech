"""Les formules de la fiche d'electromagnetisme, refaites par la bibliotheque.

Les reponses des quinze interrogations cotees sont deja tenues par les
tests de leurs corriges. Ce fichier tient ce que la fiche ajoute : les
formules de ses encadres, tapees a la main dans
`cursus/fiches/Electromagnetisme.toml`. Chacune est comparee a ce que la
bibliotheque calcule, pour qu'une coquille dans la fiche ne passe pas.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.electromagnetisme.domain.circuits.fermeture import courant_rl
from polytech.electromagnetisme.domain.magnetostatique.biot_et_savart import (
    integrale_de_biot_et_savart,
)
from polytech.electromagnetisme.domain.magnetostatique.champ_magnetique import (
    champ_d_un_conducteur_cylindrique,
)
from polytech.electromagnetisme.domain.symboles import RAYON, TEMPS

COURANT, E, R, L = sp.symbols("I E R L", positive=True)


def test_au_centre_d_une_spire_l_induction_vaut_mu0_i_sur_deux_a() -> None:
    """`B = (mu_0 I / 4 pi) x integrale`, et l'integrale vaut `2 pi / a`."""
    rayon = 2.0
    angle = sp.Symbol("phi")
    cercle = (rayon * sp.cos(angle), rayon * sp.sin(angle), sp.Integer(0))
    _, _, axiale = integrale_de_biot_et_savart(
        cercle, angle, 0.0, 2 * math.pi, (0, 0, 0)
    )
    assert math.isclose(axiale / (4 * math.pi), 1 / (2 * rayon), rel_tol=1e-9)


def test_autour_d_un_fil_le_champ_vaut_i_sur_deux_pi_r() -> None:
    """Hors du conducteur : `H = I / (2 pi r)`, par le theoreme d'Ampere."""
    dehors = champ_d_un_conducteur_cylindrique(COURANT, sp.Rational(1, 1000))
    assert sp.simplify(dehors.subs(RAYON, 3) - COURANT / (6 * sp.pi)) == 0


def test_la_fermeture_sur_une_bobine_suit_la_forme_du_premier_ordre() -> None:
    """`x(t) = x_inf + (x(0+) - x_inf) e^{-t/tau}`, `x(0+) = 0`, `tau = L/R`."""
    fiche = E / R + (0 - E / R) * sp.exp(-TEMPS / (L / R))
    assert sp.simplify(courant_rl(E, R, L) - fiche) == 0


def test_la_force_entre_deux_fils_sort_du_produit_i_l_b() -> None:
    """`F / l = I_2 B_1`, avec `B_1 = mu_0 H_1` a trois metres du premier fil."""
    mu0, i1, i2 = sp.symbols("mu_0 I_1 I_2", positive=True)
    champ = champ_d_un_conducteur_cylindrique(i1, sp.Rational(1, 1000)).subs(RAYON, 3)
    assert sp.simplify(i2 * mu0 * champ - mu0 * i1 * i2 / (2 * sp.pi * 3)) == 0
