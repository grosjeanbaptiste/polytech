"""Images, Gauss et force entre fils de la fiche d'electromagnetisme."""

from __future__ import annotations

import math

import sympy as sp

from polytech.electromagnetisme.domain.constantes import (
    PERMEABILITE_DU_VIDE as MU0,
)
from polytech.electromagnetisme.domain.electrostatique.cylindre import PERMITTIVITE
from polytech.electromagnetisme.domain.electrostatique.droites_paralleles import (
    potentiel_de_deux_droites,
)
from polytech.electromagnetisme.domain.electrostatique.sphere_uniforme import (
    deplacement_radial,
)
from polytech.electromagnetisme.domain.magnetostatique.biot_et_savart import (
    integrale_de_biot_et_savart,
)
from polytech.electromagnetisme.domain.symboles import (
    ABSCISSE,
    ORDONNEE,
)
from polytech.maths.domain.comparaison import egales, proches_relativement

B, ELL, V, X0, T = sp.symbols("B ell v x_0 T", positive=True)


def test_le_fil_image_donne_le_potentiel_de_la_fiche_et_la_charge_opposee() -> None:
    """`V = lambda/(2 pi eps_0) ln(r'/r)`, nul sur le plan ; charge `-lambda`."""
    lam, a = sp.symbols("lambda a", positive=True)
    v = potentiel_de_deux_droites(lam, a)
    r = sp.sqrt((ABSCISSE + a) ** 2 + ORDONNEE**2)  # distance au fil de charge +lambda
    r_image = sp.sqrt((ABSCISSE - a) ** 2 + ORDONNEE**2)
    assert egales(v, lam / (2 * sp.pi * PERMITTIVITE) * sp.log(r_image / r))
    assert egales(v.subs(ABSCISSE, 0), 0)
    sigma = -PERMITTIVITE * sp.diff(v, ABSCISSE).subs(ABSCISSE, 0)
    assert egales(sp.integrate(sp.simplify(sigma), (ORDONNEE, -sp.oo, sp.oo)), lam)


def test_le_deplacement_de_gauss_compte_la_charge_region_par_region() -> None:
    """`4 pi r^2 D = Q_int` : creux nul, coquille `(r^3 - R_1^3)` ; `D` continu."""
    s, r = sp.symbols("s r", positive=True)
    rho, r1 = sp.Integer(3), sp.Integer(1)
    d = deplacement_radial(sp.Piecewise((0, s < r1), (rho, True)), s, r)
    enlacee = sp.Rational(4, 3) * sp.pi * rho * (r**3 - r1**3)
    point = {r: sp.Rational(3, 2)}
    assert egales((4 * sp.pi * r**2 * d).subs(point), enlacee.subs(point))
    assert egales(d.subs(r, sp.Rational(1, 2)), 0)
    eps_r = sp.Integer(
        4
    )  # a R_2, D = eps E continu : E saute du rapport des permittivites
    champ_dedans, champ_dehors = d / (eps_r * PERMITTIVITE), d / PERMITTIVITE
    assert egales(champ_dehors / champ_dedans, eps_r)


def test_deux_fils_de_meme_sens_s_attirent_avec_la_force_de_la_fiche() -> None:
    """`F/l = I_2 B_1 = mu_0 I_1 I_2 / (2 pi d)`, dirigee vers le fil 1."""
    d, i1, i2, demi = 0.2, 3.0, 5.0, 2.0e3
    _, by, _ = (
        MU0 * i1 / (4 * math.pi) * c
        for c in integrale_de_biot_et_savart((0, 0, V), V, -demi, demi, (d, 0, 0))
    )
    force_x = -i2 * by  # z ^ y = -x : F = I_2 l ^ B_1
    assert proches_relativement(-force_x, MU0 * i1 * i2 / (2 * math.pi * d), 1e-5)
    assert force_x < 0  # le fil 2 est en x = +d : la force le tire vers le fil 1
