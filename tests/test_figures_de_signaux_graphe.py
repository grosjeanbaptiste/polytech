"""Le graphe de fluence de la figure, relu par la regle de Mason de la lib."""

from __future__ import annotations

import re

import sympy as sp

from polytech.maths.domain.comparaison import egales
from polytech.signaux.domain.fluence import regle_de_mason, transmittance_du_graphe
from polytech.signaux.domain.symboles import IMAGE as P
from tests.fiches_toml import figure

COUVERTES = {"graphes_de_fluence_et_regle_de_mason"}
A, B = sp.symbols("a b")


def test_le_graphe_de_la_figure_est_un_second_ordre_de_mason() -> None:
    texte = figure("SignauxSystemes", "graphes_de_fluence_et_regle_de_mason")
    gains = re.findall(r"node\[(?:above|below)\] \{\$([^$]*)\$\}", texte)
    assert gains == ["1", r"\frac1p", r"\frac1p", "-a", "-b"]
    parcours = [sp.Integer(1) * (1 / P) * (1 / P)]
    boucles = [-A / P, -B / P**2]
    mason = regle_de_mason(parcours, boucles)
    assert egales(mason, 1 / (P**2 + A * P + B))
    assert egales(transmittance_du_graphe(retours=[-A, -B], directs=[0, 1]), mason)
