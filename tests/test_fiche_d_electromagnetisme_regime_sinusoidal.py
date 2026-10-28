"""Le regime sinusoidal de la fiche d'electromagnetisme, refait par `polytech/`."""

from __future__ import annotations

import sympy as sp

from polytech.electromagnetisme.domain.circuits.branche_sinusoidale import (
    impedance_de_la_branche,
)
from polytech.electromagnetisme.domain.circuits.grandeur_sinusoidale import (
    GrandeurSinusoidale,
    dephasage,
    substitut_de_la_derivee,
)
from polytech.electromagnetisme.domain.circuits.phaseurs import puissance_complexe
from polytech.electromagnetisme.domain.magnetostatique.bornes_pointees import (
    flux_des_circuits_couples,
    signe_de_la_mutuelle,
)
from polytech.electromagnetisme.domain.symboles import TEMPS
from polytech.maths.domain.comparaison import dephasage_en_degres, egales, proches

E, R1, R2, R4, R, L, C, W = sp.symbols("E R_1 R_2 R_4 R L C omega", positive=True)


def test_les_substituts_complexes_de_la_fiche() -> None:
    """`E sqrt2 cos(wt + th) <-> E e^(j th)` ; `Z_R`, `Z_L = jwL`, `Z_C = 1/(jwC)`."""
    theta = sp.Rational(1, 3)
    grandeur = GrandeurSinusoidale(E, theta, W)
    assert egales(grandeur.substitut(), E * sp.exp(sp.I * theta))
    assert egales(grandeur.instantanee(), E * sp.sqrt(2) * sp.cos(W * TEMPS + theta))
    assert egales(
        substitut_de_la_derivee(grandeur.substitut(), W),
        sp.I * W * grandeur.substitut(),
    )
    assert egales(impedance_de_la_branche(R, 0, sp.oo, W), R)
    assert egales(impedance_de_la_branche(0, L, sp.oo, W), sp.I * W * L)
    assert egales(impedance_de_la_branche(0, 0, C, W), 1 / (sp.I * W * C))
    i1, i2 = 1 / (3 + 4 * sp.I), 1 / (1 + 2 * sp.I)  # meme tension, deux branches
    ecart = dephasage(sp.arg(i1), sur=sp.arg(i2))
    assert egales(ecart, sp.arg(i1 / i2))  # argument du quotient des substituts
    assert egales(
        dephasage(sp.arg(i1), sur=sp.arg(i1 * 5)), 0
    )  # quotient reel > 0 : en phase


def test_les_inductances_couplees_prennent_le_signe_des_points() -> None:
    """`u_1 = L_1 di_1 +- L_12 di_2` ; le meme `L_12` dans les deux equations."""
    l1, l2, m, i1, i2 = sp.symbols("L_1 L_2 M i_1 i_2", positive=True)
    for meme_cote, signe in ((True, 1), (False, -1)):
        flux = flux_des_circuits_couples(
            (l1, l2),
            m,
            (i1, i2),
            premier_par_le_point=True,
            second_par_le_point=meme_cote,
        )
        assert egales(flux[0], l1 * i1 + signe * m * i2)
        assert egales(flux[1], signe * m * i1 + l2 * i2)
    assert (
        signe_de_la_mutuelle(premier_par_le_point=True, second_par_le_point=False) == -1
    )


def test_des_points_inverses_retournent_le_dephasage_a_la_resonance() -> None:
    """`I_2/I_1 = -s jwM / R_2` : -90 degres (retard), +90 si `s = -1`."""
    for meme_cote, attendu in ((True, -90.0), (False, 90.0)):
        s = signe_de_la_mutuelle(
            premier_par_le_point=True, second_par_le_point=meme_cote
        )
        rapport = -s * sp.I * W * sp.Rational(1, 2) / R2
        quotient = rapport.subs({W: 3, R2: 2})
        assert proches(dephasage_en_degres(complex(quotient), 1.0), attendu, 1e-9)


def test_la_puissance_complexe_d_une_branche_rl() -> None:
    """`S = U I* = P + jQ` ; `P = R I^2`, `Q = w L I^2` pour `Z = R + jwL`."""
    courant = sp.Symbol("I", positive=True)
    z = impedance_de_la_branche(R, L, sp.oo, W)
    s = puissance_complexe(z * courant, courant)
    assert egales(sp.re(s), R * courant**2)
    assert egales(sp.im(s), W * L * courant**2)
