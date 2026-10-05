"""Syllabus d'electricite generale, chapitre 11 : le calcul des forces magnetiques.

Source : `Electromagnetisme/SYLLABUS/Syllabus_Electricite_generale.pdf`,
pages 11-1 a 11-5, lues sur les pages rendues.

Le chapitre demontre deux theoremes sans les chiffrer. L'oracle est un
systeme de deux circuits rigides dont seule la mutuelle depend de la
position : la deformation y est calculee deux fois, a flux constants et a
courants constants, et les deux forces doivent etre la meme,
`I_1 I_2 dL_12/dx`.
"""

from __future__ import annotations

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.energie_et_forces import (
    DEPLACEMENT,
    force_a_courant_constant,
    force_a_flux_constant,
)
from polytech.electromagnetisme.domain.magnetostatique.actions import (
    force_electromagnetique,
)
from polytech.electromagnetisme.domain.magnetostatique.forces_magnetiques import (
    bilan_a_courants_constants,
    couple_par_l_energie,
    force_volumique,
    variation_a_courants_constants,
    variation_a_flux_constants,
)
from polytech.electromagnetisme.domain.magnetostatique.inductances import (
    SystemeDeCircuits,
)

L1, L2, I1, I2 = sp.symbols("L_1 L_2 I_1 I_2", positive=True)
MUTUELLE = sp.Function("M")(DEPLACEMENT)
MATRICE = sp.Matrix([[L1, MUTUELLE], [MUTUELLE, L2]])
COURANTS = sp.Matrix([I1, I2])
PENTE = sp.diff(MUTUELLE, DEPLACEMENT)
FORCE = I1 * I2 * PENTE


def test_la_force_volumique_compte_les_courants_fictifs() -> None:
    """(11.1.3) : `f = (i + i') ^ B`, l'aimantation agissant par `i' = rot M`."""
    force = force_volumique((2, 0, 0), (1, 0, 0), (0, 0, 5))
    assert force == (0, -15, 0)


def test_la_force_volumique_est_celle_de_lorentz_par_unite_de_volume() -> None:
    """Une charge volumique `rho` de vitesse `v` porte le courant `rho v`."""
    charge, vitesse, induction = 3.0, (0.0, 4.0, 0.0), (0.5, 0.0, 2.0)
    courant = (charge * vitesse[0], charge * vitesse[1], charge * vitesse[2])
    lorentz = force_electromagnetique(charge, (0.0, 0.0, 0.0), vitesse, induction)
    assert force_volumique(courant, (0, 0, 0), induction) == pytest.approx(lorentz)


def test_les_deux_variations_de_l_energie_sont_opposees() -> None:
    """(11.2.5), par l'identite de Gauss : `(dW_m)_phi = -(dW_m)_I`.

    A courants constants, `d(phi) = dL . I` ; a flux constants, les
    courants varient de `dI = -L^-1 dL I`.
    """
    pas = sp.Symbol("dx")
    variation = sp.diff(MATRICE, DEPLACEMENT) * pas
    flux = SystemeDeCircuits(MATRICE).flux(COURANTS)
    a_courants = variation_a_courants_constants(COURANTS, variation * COURANTS)
    a_flux = variation_a_flux_constants(-MATRICE.inv() * variation * COURANTS, flux)
    assert sp.simplify(a_courants - FORCE * pas) == 0
    assert sp.simplify(a_flux + a_courants) == 0


def test_les_deux_theoremes_donnent_la_meme_force() -> None:
    """(11.2.11) : `F = -(dW_m/dl)_phi = +(dW_m/dl)_I`."""
    a_courants = force_a_courant_constant(SystemeDeCircuits(MATRICE).energie(COURANTS))
    assert sp.simplify(a_courants - FORCE) == 0
    phi_1, phi_2 = sp.symbols("phi_1 phi_2", real=True)
    flux = sp.Matrix([phi_1, phi_2])
    energie = (flux.T * MATRICE.inv() * flux)[0, 0] / 2
    tenus = dict(zip((phi_1, phi_2), MATRICE * COURANTS, strict=True))
    a_flux = force_a_flux_constant(energie).subs(tenus)
    assert sp.simplify(a_flux - FORCE) == 0


def test_a_courants_constants_les_sources_paient_deux_fois_le_travail() -> None:
    """(11.2.7) `dT = +(dW_m)_I`, (11.2.8) le bilan, (11.2.9) `dW_s = -2 dT`."""
    travail = sp.Symbol("dT")
    champ, sources = bilan_a_courants_constants(travail)
    assert champ == travail
    assert sources == -2 * travail
    assert travail + champ + sources == 0


def test_le_couple_derive_de_l_energie_par_rapport_a_l_angle() -> None:
    """(11.2.13) : `C = -(dW_m/d(alpha))_phi = +(dW_m/d(alpha))_I`.

    Deux bobines dont la mutuelle vaut `M_0 cos(alpha)` : le couple
    `-I_1 I_2 M_0 sin(alpha)` les ramene a `alpha = 0`, ou elles embrassent
    le plus grand flux (remarque de la page 11-4).
    """
    angle, crete = sp.Symbol("alpha", real=True), sp.Symbol("M_0", positive=True)
    energie = (L1 * I1**2 + L2 * I2**2) / 2 + crete * sp.cos(angle) * I1 * I2
    couple = couple_par_l_energie(energie, angle, a_courants_constants=True)
    assert sp.simplify(couple + I1 * I2 * crete * sp.sin(angle)) == 0
    oppose = couple_par_l_energie(energie, angle, a_courants_constants=False)
    assert sp.simplify(couple + oppose) == 0
    assert couple.subs(angle, sp.pi / 6).is_negative
