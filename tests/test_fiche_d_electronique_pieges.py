"""Fiche d'electronique : les pieges et valeurs de catalogue, portes par la lib.

Sources : `Exam_2021_2022_S2_Enonce` (Aout 2022, 3.1), `Exam_2023_2024_S1`
(4.2), `Slides_2_part1.pdf` (valeurs de catalogue de l'AOP).
"""

from __future__ import annotations

from pathlib import Path

import sympy as sp

from polytech.electronique.domain import catalogue_aop
from polytech.electronique.domain.netliste import Admittance
from polytech.electronique.domain.sonde import (
    sortie_du_non_inverseur_sonde_a_la_masse,
    transfert_avec_la_masse_de_la_sonde,
    transfert_sans_sonde,
)
from polytech.maths.domain.comparaison import egales, proches

S = sp.Symbol("s")
R, L, C = sp.symbols("R L C", positive=True)
CORPUS = Path(__file__).resolve().parent.parent / "cours-markdown"
SLIDES = CORPUS / "ElectroniqueInstrumentation" / "SLIDES" / "Slides_2_part1.md"


def _rlc() -> list[Admittance]:
    """`R` puis `L` en serie de 1 a 3, `C` en shunt de 3 a la masse."""
    return [
        Admittance(1, 2, 1 / R),
        Admittance(2, 3, 1 / (S * L)),
        Admittance(3, 0, S * C),
    ]


def test_la_masse_de_la_sonde_court_circuite_c_et_la_mesure_vaut_un() -> None:
    """Aout 2022 (3.1) : `V_m` aux bornes de `R + L`, masse sur le noeud de `C`."""
    assert egales(transfert_avec_la_masse_de_la_sonde(_rlc(), 3, (1, 3), 3), 1)
    attendu = (R + S * L) / (R + S * L + 1 / (S * C))
    assert egales(transfert_sans_sonde(_rlc(), 3, (1, 3)), attendu)


def test_la_sonde_a_la_masse_ouvre_la_contre_reaction_du_non_inverseur() -> None:
    """Janvier 2024 (4.2) : `V_- = 0` : `V_o = sat(A_v V_in)`, `-12 V` si `V_in < 0`."""
    assert sortie_du_non_inverseur_sonde_a_la_masse(-0.1, 1e5, 12.0) == -12.0
    assert sortie_du_non_inverseur_sonde_a_la_masse(0.1, 1e5, 12.0) == 12.0
    assert proches(
        sortie_du_non_inverseur_sonde_a_la_masse(1e-5, 1e5, 12.0), 1.0, 1e-12
    )


def test_les_valeurs_de_catalogue_viennent_des_planches() -> None:
    """Article « L'amplificateur » : `R_in >= 1 MOhm`, `A_v` dans `[1e4, 1e6]`."""
    texte = SLIDES.read_text(encoding="utf-8")
    regle = texte[texte.index("Good general rule") :].split("\n")[0]
    assert "1 MΩ" in regle
    plage = texte[texte.index("Typical range") :].split("\n")[0]
    assert "10<sup>4</sup>" in plage
    assert "10<sup>6</sup>" in plage
    assert catalogue_aop.R_IN_MINIMALE == 1e6
    assert catalogue_aop.GAIN_TYPIQUE == (1e4, 1e6)
    assert catalogue_aop.SOURCE.endswith("Slides_2_part1.pdf")
    assert catalogue_aop.gain_est_typique(1e5)
    assert not catalogue_aop.gain_est_typique(1e3)
