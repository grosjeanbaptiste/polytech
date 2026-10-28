"""Fiche d'electronique : cellules actives du second ordre.

Articles 26 a 29 de `cursus/fiches/ElectroniqueInstrumentation.toml` : section
passe-bas, Sallen-Key, mise en cascade, Tow-Thomas.
"""

from __future__ import annotations

import sympy as sp

from polytech.electronique.domain.biquad import TowThomas
from polytech.electronique.domain.butterworth import (
    poles_de_butterworth,
)
from polytech.electronique.domain.deux_integrateurs import tow_thomas
from polytech.electronique.domain.sallen_key_a_gain import SallenKeyPasseBas
from polytech.electronique.domain.sections_actives import biquads_de_butterworth
from polytech.maths.domain.comparaison import egales, proches

S = sp.Symbol("s")
W = sp.Symbol("w", positive=True)


def test_la_section_du_second_ordre_a_ses_poles_et_ses_gains() -> None:
    """Article « La section passe-bas » : `p = -w0/(2Q) (1 +- sqrt(1 - 4 Q^2))`."""
    g, q, w0 = sp.symbols("G Q omega_0", positive=True)
    h = g * w0**2 / (S**2 + (w0 / q) * S + w0**2)
    racines = sp.solve(sp.denom(sp.together(h)), S)
    attendu = [-w0 / (2 * q) * (1 + s * sp.sqrt(1 - 4 * q**2)) for s in (1, -1)]
    assert all(any(egales(r, a) for a in attendu) for r in racines)
    instable = sp.solve(S**2 + (1 / sp.Integer(-2)) * S + 1, S)
    assert all(sp.re(r) > 0 for r in instable)


def test_sallen_key_a_gain_porte_les_coefficients_de_la_fiche() -> None:
    """Article « Sallen-Key » : `w0`, `w0/Q` et `H(0) = mu` par l'analyse nodale."""
    r1, r2, c1, c2, mu = (sp.Rational(x) for x in ("1.2", "2.7", "0.5", "0.3", "1.5"))
    cellule = SallenKeyPasseBas(r1=r1, r2=r2, c1=c1, c2=c2, gain=mu)
    h = sp.cancel(cellule.transfert(S))
    denominateur = sp.fraction(h)[1]
    poly = sp.Poly(denominateur / sp.LC(sp.Poly(denominateur, S)), S)
    w0 = 1 / sp.sqrt(r1 * r2 * c1 * c2)
    amortissement = 1 / (r1 * c1) + 1 / (r2 * c1) + (1 - mu) / (r2 * c2)
    assert egales(poly.coeff_monomial(S), amortissement)
    assert egales(poly.coeff_monomial(1), w0**2)
    assert egales(h.subs(S, 0), mu)


def test_les_poles_conjugues_d_un_butterworth_donnent_w0_et_q_des_cellules() -> None:
    """Article « Repartir les poles » : `w0 = |p|`, `Q = |p| / (-2 Re p)`."""
    poles = [p for p in poles_de_butterworth(4) if p.imag > 0]
    coefficients = sorted(abs(p) / (abs(p) / -(2 * p.real)) for p in poles)
    assert all(
        proches(a, b, 1e-9)
        for a, b in zip(coefficients, biquads_de_butterworth(4), strict=True)
    )
    p = sp.Rational(-1, 2) + sp.I * sp.sqrt(3) / 2
    produit = sp.expand((S - p) * (S - sp.conjugate(p)))
    assert egales(produit, S**2 - 2 * sp.re(p) * S + sp.Abs(p) ** 2)


def test_tow_thomas_avec_les_resistances_de_la_fiche() -> None:
    """Article « Tow-Thomas » : `R2 = R3 = 1/w0`, `R1 = Q/w0`, `R4 = 1/(w0 G)`."""
    w0, q, gain = sp.Rational(3), sp.Rational(5, 2), sp.Rational(2)
    cellule = TowThomas(pulsation=3.0, qualite=2.5, gain=2.0)
    r1, r2, r4 = (
        sp.nsimplify(x)
        for x in (
            cellule.r1_normalisee(),
            cellule.r2_normalisee(),
            cellule.r4_normalisee(),
        )
    )
    assert egales(r1, q / w0)
    assert egales(r4, 1 / (w0 * gain))
    passe_bas, _ = tow_thomas((r1, r2, r2, r4), (1, 1), S)
    # La sortie de la planche 55 passe par trois inverseurs : le gain y est `-G`.
    assert egales(passe_bas, -gain * w0**2 / (S**2 + (w0 / q) * S + w0**2))
