"""Les champs de la fiche d'electromagnetisme, refaits par `polytech/`.

Complete `test_fiche_d_electromagnetisme.py` : induction, circuits
magnetiques, Ampere, images, Biot et Savart, Gauss, force entre fils.
"""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.constantes import (
    PERMEABILITE_DU_VIDE as MU0,
)
from polytech.electromagnetisme.domain.magnetostatique.biot_et_savart import (
    induction_d_un_element,
)
from polytech.electromagnetisme.domain.magnetostatique.champ_magnetique import (
    champ_d_un_conducteur_cylindrique,
)
from polytech.electromagnetisme.domain.magnetostatique.circuits_magnetiques import (
    flux_par_hopkinson,
    inductance_depuis_les_flux,
    reluctance,
)
from polytech.electromagnetisme.domain.magnetostatique.induction import (
    fem_de_deplacement,
    force_electromotrice,
)
from polytech.electromagnetisme.domain.magnetostatique.reluctances import (
    force_magnetomotrice,
)
from polytech.electromagnetisme.domain.symboles import (
    RAYON,
    TEMPS,
)
from polytech.maths.domain.comparaison import egales, proches_relativement

B, ELL, V, X0, T = sp.symbols("B ell v x_0 T", positive=True)


def test_la_tige_qui_glisse_donne_b_l_v() -> None:
    """`e = -dPhi/dt`, `Phi = B l x(t)` : `|e| = B l v`."""
    flux = B * ELL * (X0 + V * TEMPS)
    assert egales(force_electromotrice(flux), -B * ELL * V)
    assert proches_relativement(
        fem_de_deplacement(0.5, 0.2, 3.0), 0.5 * 0.2 * 3.0, 1e-12
    )


def test_la_spire_qui_tombe_dans_b_z_croit_en_b_pi_r2_g_t() -> None:
    """`B_z = B z`, `z = h - g t^2/2` : `|e| = B pi R^2 g t`."""
    h, g, r = sp.symbols("h g R", positive=True)
    flux = B * (h - g * TEMPS**2 / 2) * sp.pi * r**2
    assert egales(force_electromotrice(flux), B * sp.pi * r**2 * g * TEMPS)


def test_l_aire_sous_la_fem_est_le_flux_initial() -> None:
    """`e = -N dPhi/dt` jusqu'a un flux nul : `integrale e dt = N Phi(0)`."""
    n, phi0 = sp.symbols("N Phi_0", positive=True)
    flux = n * phi0 * (1 - TEMPS / T)  # le flux decroit lineairement jusqu'a zero
    aire = sp.integrate(force_electromotrice(flux), (TEMPS, 0, T))
    assert egales(aire, n * phi0)


def test_les_ampere_tours_et_hopkinson_sont_d_accord_avec_l_inductance() -> None:
    """`N I = somme H l`, `R = l/(mu S)`, `Phi = B S`, `L = N Phi / I`."""
    n, i, s, mu_r = 200, 1.5, 4e-4, 2000.0
    fer, entrefer = (0.3, mu_r), (2e-3, 1.0)
    tubes = [
        reluctance(longueur, s, relative) for longueur, relative in (fer, entrefer)
    ]
    assert proches_relativement(tubes[1], entrefer[0] / (MU0 * s), 1e-12)
    flux = flux_par_hopkinson(force_magnetomotrice([(n, i)]), sum(tubes))
    champs = [flux / s / (relative * MU0) for _, relative in (fer, entrefer)]
    somme_hl = champs[0] * fer[0] + champs[1] * entrefer[0]
    assert proches_relativement(somme_hl, n * i, 1e-9)
    assert proches_relativement(champs[1], (flux / s) / MU0, 1e-12)
    assert proches_relativement(
        inductance_depuis_les_flux(n, (flux,), i), n**2 / sum(tubes), 1e-9
    )


def test_la_circulation_de_h_autour_d_un_fil_vaut_le_courant() -> None:
    """`oint H . dl = I` : `H(r) 2 pi r = I` hors du conducteur."""
    courant, a = sp.symbols("I a", positive=True)
    champ = champ_d_un_conducteur_cylindrique(courant, a).subs(RAYON, 2 * a)
    assert egales(champ * 2 * sp.pi * 2 * a, courant)


def test_biot_et_savart_donne_un_element_perpendiculaire_en_mu0_i_sur_4_pi_r2() -> None:
    """`dB = mu_0 I/(4 pi) dl ^ r / r^3` : `x ^ y = z`, en `1/r^2`."""
    mu0, i, dl, r = sp.symbols("mu_0 I dl r", positive=True)
    bx, by, bz = induction_d_un_element(i, (dl, 0, 0), (0, r, 0))
    assert egales(sp.Matrix([bx, by]), sp.zeros(2, 1))
    assert egales(bz, mu0 * i * dl / (4 * sp.pi * r**2))
