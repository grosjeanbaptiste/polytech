"""Fiche d'electronique : prototypes de filtres, echelle LC, cellules actives.

Articles 21 a 29 de `cursus/fiches/ElectroniqueInstrumentation.toml`. Les
valeurs de composants sont verifiees par la chaine ABCD du circuit lui-meme :
trois decibels a la coupure, `1/2` en continu pour un passe-bas, `1/2` a
l'infini pour un passe-haut.
"""

from __future__ import annotations

import math

import sympy as sp

from polytech.electronique.domain import abcd
from polytech.electronique.domain.butterworth import (
    module_de_butterworth,
    poles_de_butterworth,
)
from polytech.electronique.domain.chebyshev import (
    charge_de_chebyshev,
    elements_de_chebyshev,
)
from polytech.electronique.domain.echelle_lc import (
    elements_de_butterworth,
    transfert_de_l_echelle,
)
from polytech.electronique.domain.transpositions import mise_a_l_echelle_d_impedance
from polytech.maths.domain.comparaison import proches

S = sp.Symbol("s")
W = sp.Symbol("w", positive=True)


def _gain(chaine: sp.Matrix, r0: float, omega: float) -> float:
    """`|V_charge / V_source|`, source et charge de `r0` ohms."""
    complet = abcd.cascade(abcd.serie(sp.Float(r0)), chaine)
    return abs(complex(abcd.gain_en_tension(complet, sp.Float(r0)).subs(S, 1j * omega)))


def _passe_bas(g: tuple[float, ...], r0: float, omega_c: float) -> sp.Matrix:
    """Bobines en serie, condensateurs en derivation : `L = g r0/w`, `C = g/(r0 w)`."""
    cellules = []
    for rang, valeur in enumerate(g):
        if rang % 2 == 0:
            cellules.append(abcd.serie(S * valeur * r0 / omega_c))
        else:
            cellules.append(abcd.derivation(1 / (S * valeur / (r0 * omega_c))))
    return abcd.cascade(*cellules)


def _passe_haut(g: tuple[float, ...], r0: float, omega_c: float) -> sp.Matrix:
    """Condensateurs en serie, bobines en derivation (`s -> w_c / s`)."""
    cellules = []
    for rang, valeur in enumerate(g):
        if rang % 2 == 0:
            cellules.append(abcd.serie(valeur * r0 * omega_c / S))
        else:
            cellules.append(abcd.derivation(S * r0 / (valeur * omega_c)))
    return abcd.cascade(*cellules)


def test_butterworth_poles_sur_un_cercle_et_trois_decibels_a_la_coupure() -> None:
    """Article « Reconnaitre un Butterworth » : `|p_k| = w_p / eps^(1/N)`."""
    for ordre, epsilon in ((3, 1.0), (4, 0.5), (5, 0.8)):
        modules = {
            round(abs(p), 9) for p in poles_de_butterworth(ordre, epsilon=epsilon)
        }
        assert modules == {round(epsilon ** (-1 / ordre), 9)}
    coupure = abs(poles_de_butterworth(4)[0])
    assert proches(module_de_butterworth(coupure, 4), 1 / math.sqrt(2), 1e-12)


def test_l_echelle_de_butterworth_est_la_moitie_de_butterworth() -> None:
    """Article « Le prototype en echelle LC » : `g_i = 2 sin((2i-1) pi / 2N)`."""
    for ordre in (2, 3, 4):
        g = elements_de_butterworth(ordre)
        attendu = tuple(
            2 * math.sin((2 * i - 1) * math.pi / (2 * ordre))
            for i in range(1, ordre + 1)
        )
        assert g == attendu or all(
            proches(a, b, 1e-12) for a, b in zip(g, attendu, strict=True)
        )
        h = transfert_de_l_echelle(g, S)
        for omega in (0.5, 1.0, 2.0):
            module = abs(complex(h.subs(S, 1j * omega)))
            assert proches(module, module_de_butterworth(omega, ordre) / 2, 1e-9)


def test_la_charge_de_chebyshev_aux_ordres_pairs_est_coth_carre() -> None:
    """Piege de l'article : `1,3554` a `0,1` dB, `coth^2(beta/4)`, pas `coth`."""
    beta = math.log(1 / math.tanh(0.1 / 17.37))
    assert proches(charge_de_chebyshev(0.1, 2), 1.3554, 1e-4)
    assert proches(charge_de_chebyshev(0.1, 2), (1 / math.tanh(beta / 4)) ** 2, 2e-4)
    assert not proches(charge_de_chebyshev(0.1, 2), 1 / math.tanh(beta / 4), 0.1)
    assert charge_de_chebyshev(0.1, 3) == 1
    g = elements_de_chebyshev(0.1, 3)
    assert proches(g[0], 1.0316, 1e-4)
    assert proches(g[1], 1.1474, 1e-4)


def test_butterworth_d_ordre_2_a_400_mhz_sur_73_ohms() -> None:
    """Janvier 2024 (2.2) : bobine `41,08 nH`, capacite `7,71 pF` ; trois dB a `f_c`."""
    omega = 2 * math.pi * 400e6
    g = elements_de_butterworth(2)
    bobine, capacite = g[0] * 73 / omega, g[1] / (73 * omega)
    assert proches(bobine, 41.08e-9, 5e-12)
    assert proches(capacite, 7.71e-12, 5e-15)
    chaine = _passe_bas(g, 73.0, omega)
    assert proches(_gain(chaine, 73.0, omega), 0.5 / math.sqrt(2), 1e-9)
    assert proches(_gain(chaine, 73.0, omega / 1e4), 0.5, 1e-6)


def test_le_passe_haut_d_ordre_3_a_50_mhz_sur_50_ohms() -> None:
    """Janvier 2025 (2.2) : `C = 1/(g R0 w_c)`, `L = R0/(g w_c)` ; `1/2` a l'infini."""
    omega = 2 * math.pi * 50e6
    g = elements_de_butterworth(3)
    assert proches(1 / (g[0] * 50 * omega), 63.66e-12, 5e-15)
    assert proches(50 / (g[1] * omega), 79.58e-9, 5e-11)
    chaine = _passe_haut(g, 50.0, omega)
    assert proches(_gain(chaine, 50.0, omega), 0.5 / math.sqrt(2), 1e-9)
    assert proches(_gain(chaine, 50.0, omega * 1e4), 0.5, 1e-6)
    assert proches(_gain(chaine, 50.0, omega / 1e4), 0.0, 1e-9)


def test_la_mise_a_l_echelle_d_impedance_laisse_le_transfert_inchange() -> None:
    """Article « La mise a l'echelle » : `R -> k R`, `L -> k L`, `C -> C / k`."""
    resistance, bobine, capacite = mise_a_l_echelle_d_impedance(0.5, 90, 1e-6, 1e-9)
    omega = 2e6
    avant = abcd.cascade(abcd.serie(S * 1e-6), abcd.derivation(1 / (S * 1e-9)))
    apres = abcd.cascade(abcd.serie(S * bobine), abcd.derivation(1 / (S * capacite)))
    assert proches(_gain(avant, 90.0, omega), _gain(apres, resistance, omega), 1e-9)
