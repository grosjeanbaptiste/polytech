"""Seconds chemins pour les lectures de planches de fluides, depuis le TEXTE converti.

Lecture seule de `cours-markdown/`. Une valeur qui n'est que dans une
figure reste « non verifiable sans le scan ».
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

SLIDES = (
    Path(__file__).resolve().parent.parent / "cours-markdown/MecaniqueFluides/SLIDES"
)


def _texte(nom: str) -> str:
    chemin = SLIDES / nom
    if not chemin.exists():
        pytest.skip(f"texte converti absent : {chemin}")
    return chemin.read_text(encoding="utf-8")


def test_planche_12_le_texte_imprime_un_denominateur_sans_h_carre() -> None:
    """Classe (c) : texte = `(h0^2 - hL^2)` seul ; planche 14 : 10^7 Pa."""
    texte = _texte("Slides_07_Lubrification_Ecoulement_turbulent.md")
    ligne = next(m for m in texte.splitlines() if "p ( h ) − p 0" in m)
    imprime = re.search(
        r"6 μUL \( h 0 − h \)\( h  − hL \)<br>\( h 0 2  − hL 2 \)", ligne
    )
    assert imprime, "formule de la planche 12 absente du texte"
    assert "10<sup>7</sup>" in texte or "107 Pascal" in texte
    mu, u, longueur, h0, hl = 0.1, 10.0, 0.1, 0.1e-3, 0.05e-3
    h = 2 * h0 * hl / (h0 + hl)
    sans = 6 * mu * u * longueur * (h0 - h) * (h - hl) / (h0**2 - hl**2)
    avec = sans / h**2
    assert sans == pytest.approx(0.0444, abs=5e-5)
    assert avec == pytest.approx(1e7)


def test_planche_22_le_texte_imprime_une_paire_d_efforts_non_orthogonale() -> None:
    """Classe (c) : (4.69)/(4.70) imprimes donnent un frottement non tangent."""
    texte = _texte("Slides_08_Aerodynamique.md")
    assert "dFy = − p sin(θ)dA + τw cos(θ) dA" in texte
    assert "dFx = − p cos(θ)dA + τw sin(θ) dA" in texte
    t = 0.7
    normale = (math.cos(t), math.sin(t))
    imprime = (math.sin(t), math.cos(t))  # coefficients de tau_w imprimes (x, y)
    notre = (math.sin(t), -math.cos(t))
    assert sum(a * b for a, b in zip(normale, imprime, strict=True)) == pytest.approx(
        math.sin(2 * t)
    )
    assert sum(a * b for a, b in zip(normale, notre, strict=True)) == pytest.approx(0.0)


def test_planche_9_texte_donne_110_mpa_et_1100_atm() -> None:
    """Classe (a) : `1000 x 10 x 11000`, `1100 atm` imprimes."""
    texte = _texte("Slides_02_Statique_Cinematique.md")
    assert "1000 × 10 × 11000" in texte
    assert "1100 atm" in texte
    assert 1000 * 10 * 11000 == 110_000_000
