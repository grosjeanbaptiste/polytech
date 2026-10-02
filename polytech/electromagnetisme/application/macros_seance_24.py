"""Macros de la seance 2.4, exercices 2, 3, 4 et 9 : les litteraux.

Chaque generateur equivalent est publie par les **trois mesures** du
domaine -- tension a vide, resistance du reseau eteint, courant de
court-circuit --, qui sont trois resolutions distinctes par la loi des
noeuds. Aucune ne reprend la loi des mailles du manuscrit : que les
expressions coincident avec les siennes est ce que le document montre.

Les courants de Norton sont publies **developpes** : ils se lisent alors
terme a terme sur le schema, ce que la forme reduite au meme
denominateur du manuscrit cache.

Les prefixes sont `DeuxQuatreDeux`, `DeuxQuatreTrois`, `DeuxQuatreQuatre`
et `DeuxQuatreNeuf`. L'exercice 6, numerique, est dans
`macros_seance_24_six`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

from polytech.electromagnetisme.application.enonces_seance_24 import (
    reseau_du_deux,
    reseau_du_neuf,
    reseau_du_quatre,
    reseau_du_trois,
)
from polytech.electromagnetisme.application.formatage import expression_latex
from polytech.electromagnetisme.application.macros_seance_24_six import (
    macros_du_six,
)
from polytech.electromagnetisme.application.symboles_seance_24 import (
    CHARGE,
    NUMERO_DE_LA_SEANCE,
    R1,
)
from polytech.electromagnetisme.domain.circuits.equivalents import (
    courant_de_court_circuit,
    resistance_vue_des_bornes,
    tension_a_vide,
)

if TYPE_CHECKING:
    from polytech.electromagnetisme.domain.circuits.reseau import Reseau


def _generateur(prefixe: str, reseau: Reseau) -> dict[str, str]:
    """La tension, la resistance et le courant de Norton d'un reseau."""
    norton = courant_de_court_circuit(reseau, "A", "B")
    return {
        f"{prefixe}Tension": expression_latex(
            sp.together(tension_a_vide(reseau, "A", "B")),
        ),
        f"{prefixe}Resistance": expression_latex(
            sp.together(resistance_vue_des_bornes(reseau, "A", "B")),
        ),
        f"{prefixe}Norton": expression_latex(sp.apart(sp.together(norton), R1)),
    }


def _macros_du_deux() -> dict[str, str]:
    """Le generateur, puis le courant dans la charge par deux routes."""
    nu = reseau_du_deux()
    a_vide = tension_a_vide(nu, "A", "B")
    resistance = resistance_vue_des_bornes(nu, "A", "B")
    complet = reseau_du_deux(charge=True).potentiels()["A"] / CHARGE
    return {
        **_generateur("DeuxQuatreDeux", nu),
        "DeuxQuatreDeuxCourant": expression_latex(
            sp.factor(sp.simplify(a_vide / (resistance + CHARGE))),
        ),
        "DeuxQuatreDeuxCourantDirect": expression_latex(
            sp.factor(sp.simplify(complet)),
        ),
    }


def _macros_du_neuf() -> dict[str, str]:
    """Le generateur de Norton, et la tension nulle aux bornes de `R_1`."""
    reseau = reseau_du_neuf()
    potentiels = reseau.potentiels()
    return {
        **_generateur("DeuxQuatreNeuf", reseau),
        "DeuxQuatreNeufTensionSurRUn": expression_latex(
            sp.simplify(potentiels["m"] - potentiels["A"]),
        ),
    }


def macros_de_la_seance_deux_quatre() -> dict[str, str]:
    """Les cinq exercices de la seance : quatre litteraux, un numerique."""
    return {
        "DeuxQuatreSeance": NUMERO_DE_LA_SEANCE,
        **_macros_du_deux(),
        **_generateur("DeuxQuatreTrois", reseau_du_trois()),
        **_generateur("DeuxQuatreQuatre", reseau_du_quatre()),
        **macros_du_six(),
        **_macros_du_neuf(),
    }
