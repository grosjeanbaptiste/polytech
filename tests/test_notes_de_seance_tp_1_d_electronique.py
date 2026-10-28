"""Les notes de seance du TP 1 d'electronique, recalculees.

Source : `ElectroniqueInstrumentation/TP/TP_1_Notes_de_seance_2026_2027.pdf`,
scan manuscrit de onze pages, transcrit a la main dans
`transcriptions/ElectroniqueInstrumentation/TP/`.

Aucun nombre n'est recopie pour etre compare a lui-meme : chacun est
refait par `polytech.electronique` ou par sympy, et la transcription doit
porter la valeur que ce second chemin donne.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import sympy as sp

from polytech.electromagnetisme.domain.constantes import PERMEABILITE_DU_VIDE
from polytech.electronique.domain.laplace import (
    diviseur_de_tension,
    impedances_en_parallele,
    impedances_en_serie,
)
from polytech.electronique.domain.solenoide import (
    courant_de_saturation,
    frequence_de_croisement,
    inductance_de_rosa,
    inductance_du_solenoide,
    permeabilite_lineaire,
)

RACINE = Path(__file__).resolve().parent.parent
NOTES = (
    RACINE
    / "transcriptions/ElectroniqueInstrumentation/TP"
    / "TP_1_Notes_de_seance_2026_2027.md"
)
MU, N, L, A, D, T, W, B0 = sp.symbols("mu N l A d t omega B_0", positive=True)
COURANT = sp.Function("I")


@pytest.fixture(scope="module")
def texte() -> str:
    return NOTES.read_text(encoding="utf-8")


def test_l_inductance_du_solenoide_vient_de_faraday() -> None:
    """`V = N V_b` et `V_b = mu N/l A dI/dt` : `L = mu N^2 A / l`."""
    v_b = sp.diff(MU * N / L * A * COURANT(T), T)
    inductance = sp.simplify(N * v_b / sp.diff(COURANT(T), T))
    assert inductance == MU * N**2 * A / L
    assert sp.simplify(inductance.subs(L, N * D) - MU * N * A / D) == 0


def test_le_ferrite_de_la_datasheet(texte: str) -> None:
    """`mu ~ 9e-4`, `mu_r ~ 750`, `I_max = 160 mA`, `L = 450 uH` arrondi."""
    mu = permeabilite_lineaire(0.15, 160.0)
    assert round(mu, 4) == pytest.approx(9e-4)
    assert round(mu / PERMEABILITE_DU_VIDE, -1) == pytest.approx(750)
    assert courant_de_saturation(160.0, 20 * 1e-3, 20) == pytest.approx(0.160)
    assert inductance_du_solenoide(9e-4, 20, 25e-6, 20e-3) == pytest.approx(450e-6)
    for ecrit in ("9 \\times 10^{-4}", "\\simeq 750", "160\\ \\text{mA}", "450\\ \\mu"):
        assert ecrit in texte


def test_la_tension_de_la_boucle_est_moins_la_derivee_du_flux() -> None:
    """Exercice 3.1 : `V_R = -d/dt (B_0 sin(omega t) A)`."""
    v_r = -sp.diff(B0 * sp.sin(W * T) * A, T)
    assert sp.simplify(v_r + B0 * A * W * sp.cos(W * T)) == 0


def test_l_inductance_de_rosa_et_les_deux_croisements(texte: str) -> None:
    """153,28 nH, puis 10,38 MHz et 10,38 THz.

    La formule de Rosa donne 153,272 nH ; le manuscrit ecrit 153,28, ce
    que l'on retrouve en arrondissant `ln(a/rho)` au millieme avant de
    multiplier. Le manuscrit ecrit bien des nanohenrys -- lu agrandi, a
    cote du « mA » de la page 4 --, la ou l'officiel ecrit « mH ».
    """
    inductance = inductance_de_rosa(0.05, 0.65e-3)
    assert round(inductance * 1e9, 2) == pytest.approx(153.27)
    arrondi = inductance_de_rosa(0.05, 0.65e-3, decimales_du_logarithme=3)
    assert arrondi * 1e9 == pytest.approx(153.28, abs=0.005)
    assert round(arrondi * 1e9, 2) == pytest.approx(153.28)
    assert round(frequence_de_croisement(10.0, inductance) / 1e6, 2) == 10.38
    assert round(frequence_de_croisement(10e6, inductance) / 1e12, 2) == 10.38
    for ecrit in (
        "153{,}28\\ \\text{nH}",
        "10{,}38\\ \\text{MHz}",
        "10{,}38\\ \\text{THz}",
    ):
        assert ecrit in texte


def test_le_pont_de_resistances(texte: str) -> None:
    """`R_1 // R_2 = R_1/2`, `Z_eq = 5,5 kOhm`, `V_a = 50/11 V ~ 4,55 V`."""
    r_1 = sp.Symbol("R_1", positive=True)
    assert sp.simplify(impedances_en_parallele(r_1, r_1) - r_1 / 2) == 0
    amont = impedances_en_parallele(sp.Integer(1000), sp.Integer(1000))
    aval = impedances_en_parallele(sp.Integer(10_000), sp.Integer(10_000))
    assert impedances_en_serie(amont, aval) == 5500
    tension = 5 * diviseur_de_tension(amont, aval)
    assert tension == sp.Rational(50, 11)
    assert round(float(tension), 2) == 4.55
    assert "4{,}55\\ \\text{V}" in texte


def test_la_frequence_croit_comme_la_resistance() -> None:
    """De 10 ohms a 10 megohms, le facteur `10^6` que le manuscrit ecrit."""
    inductance = inductance_de_rosa(0.05, 0.65e-3)
    rapport = frequence_de_croisement(10e6, inductance) / frequence_de_croisement(
        10.0, inductance
    )
    assert math.isclose(rapport, 1e6)
