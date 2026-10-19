"""Les notes de seance du TP 2 d'electronique, recalculees.

Source : `ElectroniqueInstrumentation/TP/TP_2_Notes_de_seance_2026_2027.pdf`,
scan manuscrit de huit pages, transcrit a la main dans
`transcriptions/ElectroniqueInstrumentation/TP/`.

Chaque impedance, transfert et superposition est refait par
`polytech.electronique.domain.laplace` ; la superposition est recoupee par
l'analyse nodale, un chemin que le manuscrit ne prend pas. C'est elle qui
dement le facteur 2 du total, que la transcription signale.
"""

from __future__ import annotations

import cmath
import math
from pathlib import Path

import sympy as sp

from polytech.electronique.domain.laplace import (
    diviseur_de_tension,
    impedance_de_bobine,
    impedance_de_condensateur,
    impedance_de_resistance,
    impedances_en_parallele,
    impedances_en_serie,
)

RACINE = Path(__file__).resolve().parent.parent
NOTES = (
    RACINE
    / "transcriptions/ElectroniqueInstrumentation/TP"
    / "TP_2_Notes_de_seance_2026_2027.md"
)
S = sp.Symbol("s")
W = sp.Symbol("omega", positive=True)
R, L, C, C_L, R_L, R_S, R_1, R_2 = sp.symbols(
    "R L C C_L R_L R_S R_1 R_2", positive=True
)
Z, Z_1, Z_2, Z_3 = sp.symbols("Z Z_1 Z_2 Z_3")
V_1, V_2, I_1, I_2 = sp.symbols("V_1 V_2 I_1 I_2")


def _nul(expression: sp.Expr) -> bool:
    return bool(sp.simplify(expression) == 0)


def test_trois_impedances_en_parallele() -> None:
    """La forme juste, et celle que le manuscrit barre."""
    eq = impedances_en_parallele(Z_1, Z_2, Z_3)
    assert _nul(eq - Z_1 * Z_2 * Z_3 / (Z_2 * Z_3 + Z_1 * Z_2 + Z_1 * Z_3))
    assert not _nul(eq - Z_1 * Z_2 * Z_3 / (Z_1 + Z_2 + Z_3))


def test_les_deux_1_ports_de_l_exercice_2_1() -> None:
    """`R / (sCR + 1)`, pole `-1/CR` ; `sL + R`, zero `-R/L`."""
    premier = impedances_en_parallele(
        impedance_de_condensateur(C, S), impedance_de_resistance(R, S)
    )
    assert _nul(premier - R / (S * C * R + 1))
    assert sp.solve(sp.denom(sp.together(premier)), S) == [-1 / (C * R)]
    assert sp.limit(premier, S, sp.oo) == 0
    second = impedances_en_serie(
        impedance_de_resistance(R, S), impedance_de_bobine(L, S)
    )
    assert sp.solve(second, S) == [-R / L]


def test_l_exercice_2_2() -> None:
    """La forme generale, ses zeros, et leur approximation imaginaire."""
    total = impedances_en_serie(
        impedance_de_bobine(L, S),
        impedances_en_parallele(
            impedance_de_resistance(R, S), impedance_de_condensateur(C, S)
        ),
    )
    assert _nul(total - (S**2 * L * R * C + S * L + R) / (S * R * C + 1))
    assert total.subs(S, 0) == R
    zeros = set(sp.solve(S**2 * L * R * C + S * L + R, S))
    racine = sp.sqrt(1 / (4 * R**2 * C**2) - 1 / (L * C))
    assert {sp.simplify(z) for z in zeros} == {
        sp.simplify(-1 / (2 * R * C) + racine),
        sp.simplify(-1 / (2 * R * C) - racine),
    }


def test_les_diviseurs_de_tension() -> None:
    """1,2 V donne 1 V ; la charge `Z // R_L`, puis `|Z| >> R_L`."""
    assert (
        sp.Rational(6, 5) * diviseur_de_tension(sp.Integer(1000), sp.Integer(5000)) == 1
    )
    charge = impedances_en_parallele(Z, R_L)
    transfert = diviseur_de_tension(R_S, charge)
    assert _nul(transfert - Z * R_L / (Z * R_L + Z * R_S + R_L * R_S))
    assert _nul(sp.limit(transfert, Z, sp.oo) - R_L / (R_S + R_L))


def test_la_superposition_dement_le_facteur_2() -> None:
    """La loi des noeuds en `V_out` donne `R_2 (V_1 + V_2) / (R_1 + 2 R_2)`."""
    v = sp.Symbol("V_out")
    nodal = sp.solve((v - V_1) / R_1 + (v - V_2) / R_1 + v / R_2, v)[0]
    une = diviseur_de_tension(R_1, impedances_en_parallele(R_1, R_2))
    assert _nul(une - R_1 * R_2 / (R_1**2 + 2 * R_1 * R_2))
    assert _nul(une * (V_1 + V_2) - nodal)
    assert not _nul(2 * une * (V_1 + V_2) - nodal)
    assert "sans le facteur 2" in NOTES.read_text(encoding="utf-8")


def test_les_deux_sources_de_courant() -> None:
    """Les tensions des trois noeuds, par l'analyse nodale."""
    v1, v2, v3 = sp.symbols("v1 v2 v3")
    y = S * C
    solution = sp.solve(
        [
            (v1 - v2) / R_1 - I_1,
            (v2 - v1) / R_1 + (v2 - v3) / R_2 - I_2,
            (v3 - v2) / R_2 + y * v3,
        ],
        [v1, v2, v3],
    )
    branche = R_2 + 1 / (S * C)
    assert _nul(solution[v1] - (branche * (I_1 + I_2) + R_1 * I_1))
    assert _nul(solution[v2] - branche * (I_1 + I_2))
    assert _nul(solution[v3] - (I_1 + I_2) / (S * C))


def test_les_deux_filtres_de_l_exercice_6_1() -> None:
    """Le passe-haut `sRC / (sRC + 1)`, puis le passe-bas charge."""
    haut = diviseur_de_tension(
        impedance_de_condensateur(C, S), impedance_de_resistance(R, S)
    )
    assert _nul(haut - S * R * C / (S * R * C + 1))
    reponse = haut.subs(S, sp.I * W)
    assert _nul(sp.Abs(reponse) - W * R * C / sp.sqrt(1 + W**2 * R**2 * C**2))
    for valeur in (0.1, 1.0, 10.0):
        phase = cmath.phase(complex(reponse.subs({W: valeur, R: 1, C: 1})))
        assert math.isclose(phase, math.pi / 2 - math.atan(valeur))
    c_t = impedances_en_parallele(
        impedance_de_condensateur(C, S), impedance_de_condensateur(C_L, S)
    )
    assert _nul(c_t - impedance_de_condensateur(C + C_L, S))
    bas = diviseur_de_tension(R, impedances_en_parallele(R_L, c_t)).subs(S, sp.I * W)
    module = R_L / sp.sqrt((R + R_L) ** 2 + W**2 * R**2 * (C + C_L) ** 2 * R_L**2)
    assert _nul(sp.Abs(bas) - module)
    point = {W: 3.0, R: 2, C: 0.5, C_L: 0.25, R_L: 5}
    phase = cmath.phase(complex(bas.subs(point)))
    assert math.isclose(phase, -math.atan(3.0 * 2 * 0.75 * 5 / 7))
