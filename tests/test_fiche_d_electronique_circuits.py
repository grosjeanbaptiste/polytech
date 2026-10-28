"""Fiche d'electronique : diviseurs, impedances, quadripoles, Laplace.

Articles 1 a 10 de `cursus/fiches/ElectroniqueInstrumentation.toml`, refaits
par `polytech.electronique` et `polytech.maths`, compares par `egales`.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.electronique.domain import abcd, laplace
from polytech.electronique.domain.amplification import diviseur_d_entree
from polytech.electronique.domain.conversions_de_quadripole import Jeu, convertir
from polytech.electronique.domain.netliste import (
    Admittance,
    Element,
    SourceDeCourant,
    SourceDeTension,
    SourceDeTensionCommandee,
    resoudre,
)
from polytech.electronique.domain.retroaction import parametres_h_amplificateur
from polytech.maths.domain.comparaison import egales, proches
from polytech.maths.domain.laplace import proprietes
from polytech.maths.domain.symboles import IMAGE

S = sp.Symbol("s")
R, C, L, RIN, RO, AV = sp.symbols("R C L R_in R_o A_v", positive=True)
T = sp.Symbol("t", positive=True)


def test_la_cellule_rc_et_sa_coupure() -> None:
    """Article « Le diviseur » : `1/(sRC+1)`, `f_c = 1/(2 pi RC)`."""
    zc = laplace.impedance_de_condensateur(C, S)
    h = laplace.diviseur_de_tension(laplace.impedance_de_resistance(R, S), zc)
    assert egales(h, 1 / (S * R * C + 1))
    assert egales(sp.Abs(h.subs(S, sp.I / (R * C))), 1 / sp.sqrt(2))


def test_le_condensateur_de_5_khz_sur_1_kilohm_et_les_deux_filtres_de_440_hz() -> None:
    """Janvier 2019 (3.3) et janvier 2020 (3.1) : `C = 1/(2 pi f_c R)`, `omega_c`."""
    condensateur = float(1 / (2 * sp.pi * 5e3 * 1e3))
    assert proches(condensateur, 31.83e-9, 1e-11)
    omega = 2 * math.pi * 440
    r_rc = 1 / (omega * 100e-9)
    h_rc = laplace.diviseur_de_tension(r_rc, 1 / (S * 100e-9))
    assert proches(abs(complex(h_rc.subs(S, 1j * omega))), 1 / math.sqrt(2), 1e-9)
    bobine = 50 / omega
    h_rl = laplace.diviseur_de_tension(50, S * bobine)
    assert proches(abs(complex(h_rl.subs(S, 1j * omega))), 1 / math.sqrt(2), 1e-9)
    assert proches(diviseur_d_entree(1e3, 9e3), 0.9, 1e-12)


def test_les_limites_du_condensateur_et_de_la_bobine() -> None:
    """Article « Les impedances » : `C` ouvert, `L` fil en continu."""
    zc, zl = laplace.impedance_de_condensateur(C, S), laplace.impedance_de_bobine(L, S)
    assert sp.limit(zc, S, 0, "+") == sp.oo
    assert sp.limit(zc, S, sp.oo) == 0
    assert (sp.limit(zl, S, 0), sp.limit(zl, S, sp.oo)) == (0, sp.oo)
    serie = laplace.impedances_en_serie(R, zc, zl)
    assert egales(serie, R + 1 / (S * C) + S * L)
    assert egales(1 / laplace.impedances_en_parallele(R, zc), 1 / R + S * C)


def test_la_matrice_nodale_donne_le_diviseur_et_le_courant_de_la_source() -> None:
    """Article « Analyse nodale modifiee » : la source de tension a son courant."""
    vin, i1 = sp.symbols("V_in I_1")
    circuit: list[Element] = [
        SourceDeTension(1, 0, vin, i1),
        Admittance(1, 2, 1 / R),
        Admittance(2, 0, S * C),
    ]
    v1, v2, courant = resoudre(circuit, 2)
    assert egales(v2, vin / (1 + S * R * C))
    assert egales(v1, vin)
    assert egales(courant, v2 * S * C)


def test_la_matrice_h_de_l_amplificateur_vient_de_l_analyse_nodale() -> None:
    """Article « Les parametres h » : `[[R_in, 0], [-A_v R_in/R_o, 1/R_o]]`."""
    interne = sp.Symbol("I_x")
    courants = sp.symbols("i_1 i_2")
    circuit: list[Element] = [
        Admittance(1, 0, 1 / RIN),
        SourceDeTensionCommandee(3, 0, AV, (1, 0), interne),
        Admittance(3, 2, 1 / RO),
        SourceDeCourant(0, 1, courants[0]),
        SourceDeCourant(0, 2, courants[1]),
    ]
    tensions = resoudre(circuit, 3)
    z = sp.Matrix(
        [[sp.diff(tensions[a], c) for c in courants] for a in (0, 1)]
    ).applyfunc(sp.simplify)
    h = convertir(z, depuis=Jeu.Z, vers=Jeu.H)
    attendu = sp.Matrix([[RIN, 0], [-AV * RIN / RO, 1 / RO]])
    assert egales(h, attendu)
    assert egales(parametres_h_amplificateur(RIN, RO, AV), attendu)
    assert egales(convertir(h, depuis=Jeu.H, vers=Jeu.G), h.inv())


def test_la_chaine_abcd_serie_shunt_et_la_charge() -> None:
    """Article « ABCD » : `T = T1 T2`, `V2/V1 = Z_L/(A Z_L + B)`, `Z_in`."""
    z1, z2 = sp.symbols("Z_1 Z_2", positive=True)
    zl = sp.Symbol("Z_L", positive=True)
    t = abcd.cascade(abcd.serie(z1), abcd.derivation(z2))
    assert egales(t, sp.Matrix([[1, z1], [0, 1]]) * sp.Matrix([[1, 0], [1 / z2, 1]]))
    assert egales(abcd.gain_en_tension(t, zl), zl / (t[0, 0] * zl + t[0, 1]))
    assert egales(sp.limit(abcd.gain_en_tension(t, zl), zl, sp.oo), 1 / t[0, 0])
    diviseur = laplace.diviseur_de_tension(z1, laplace.impedances_en_parallele(z2, zl))
    assert egales(abcd.gain_en_tension(t, zl), diviseur)
    zin = abcd.impedance_d_entree(t, zl)
    assert egales(zin, z1 + laplace.impedances_en_parallele(z2, zl))


def test_fractions_simples_et_reponse_a_l_echelon_du_premier_ordre() -> None:
    """Article « Revenir au temps » : `G beta / alpha (1 - e^{-alpha t})`."""
    alpha, g, beta = sp.symbols("alpha G beta", positive=True)
    y = laplace.reponse_a_un_echelon(g / (S + alpha), beta, S, T)
    assert egales(y, g * beta / alpha * (1 - sp.exp(-alpha * T)))
    r = laplace.residu_en_un_pole(g * beta / (S * (S + alpha)), S, -alpha)
    assert egales(r, -g * beta / alpha)
    assert laplace.est_stable(g / (S + alpha), S)


def test_les_trois_proprietes_de_laplace() -> None:
    """Article « Trois proprietes » : `F/s`, `sF - f(0)`, `F(s - alpha)`."""
    f = 1 / (IMAGE + 2)
    assert egales(proprietes.de_l_integrale(f), f / IMAGE)
    assert egales(proprietes.de_la_derivee_n(f, [1]), IMAGE * f - 1)
    assert egales(proprietes.du_produit_par_une_exponentielle(f, 3), 1 / (IMAGE - 1))
