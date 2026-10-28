"""Le premier inverse de l'exercice 8 vient d'une macro, pas de la frappe.

Le corrige tapait `1/10^{-2} = 100` : la distance et son inverse sont
desormais calcules par la bibliotheque, a partir de la premiere demi-largeur.
"""

from __future__ import annotations

from pathlib import Path

from polytech.electromagnetisme.application.macros_champ_08 import macros_champ_huit
from polytech.electromagnetisme.application.macros_champ_08_sommes import (
    distances_a_l_axe,
)
from polytech.maths.domain.comparaison import egales

RACINE = Path(__file__).resolve().parent.parent
CORRIGE = RACINE / "travaux/Electromagnetisme/exs-03-champ-magnetique-08"


def test_le_premier_inverse_est_celui_de_la_premiere_distance_en_metres() -> None:
    """`1 / 10^-2 m = 100 m^-1` : la premiere demi-largeur est un centimetre."""
    macros = macros_champ_huit()
    premiere = distances_a_l_axe()[0]
    assert macros["ChampHuitPremiereDistanceEnMetres"] == "0.01"
    assert egales(float(macros["ChampHuitPremierInverse"]), 1 / premiere)
    assert macros["ChampHuitPremierInverse"] == "100"


def test_le_corrige_n_ecrit_plus_l_inverse_a_la_main() -> None:
    """Le `100` du corrige vient de la macro `ChampHuitPremierInverse`."""
    tex = (CORRIGE / "exs_03_champ_magnetique_08.tex").read_text(encoding="utf-8")
    assert "1/10^{-2} = 100" not in tex
    assert "\\ChampHuitPremierInverse" in tex
