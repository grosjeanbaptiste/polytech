"""Macros de l'exercice cote du 19 mai 2025 (Mons). Prefixe `CoteMonsVingtCinq`."""

from __future__ import annotations

from polytech.electromagnetisme.application import cote_2025_mons_electroaimant as el
from polytech.electromagnetisme.application import cote_2025_mons_thevenin as th
from polytech.electromagnetisme.application.formatage import (
    expression_latex,
    nombre,
)

ARRONDI = 4.8
"""Le courant que le manuscrit retient, en amperes, pour chiffrer la force."""


def _thevenin() -> dict[str, str]:
    tension, resistance = th.thevenin()
    return {
        "CoteMonsVingtCinqResistance": expression_latex(resistance),
        "CoteMonsVingtCinqTension": nombre(float(tension), 1),
        "CoteMonsVingtCinqCourantUn": nombre(float(th.courant_dans_r2()), 1),
        "CoteMonsVingtCinqSortie": nombre(float(th.tension_sur_la_resistance()), 1),
    }


def _electroaimant() -> dict[str, str]:
    hipersil, toles, air = el.partage()
    return {
        "CoteMonsVingtCinqFlux": nombre(el.flux() * 1e4, 1),
        "CoteMonsVingtCinqHipersil": nombre(hipersil, 2),
        "CoteMonsVingtCinqToles": nombre(toles, 1),
        "CoteMonsVingtCinqAir": nombre(air, 1),
        "CoteMonsVingtCinqPartDeLAir": nombre(100 * air / sum(el.partage()), 1),
        "CoteMonsVingtCinqCourant": nombre(el.courant(), 2),
        "CoteMonsVingtCinqArrondi": nombre(ARRONDI, 1),
        "CoteMonsVingtCinqApprochee": nombre(el.induction_approchee(ARRONDI), 3),
        "CoteMonsVingtCinqForce": nombre(el.force(ARRONDI), 1),
        "CoteMonsVingtCinqForceExacte": nombre(el.force(el.courant()), 1),
        "CoteMonsVingtCinqPoids": nombre(el.poids(), 3),
        "CoteMonsVingtCinqMinimal": nombre(el.courant_minimal(), 2),
    }


def macros_cote_mons_vingt_cinq() -> dict[str, str]:
    """Les macros des questions 1 et 2 ; la question 3 est litterale."""
    return {**_thevenin(), **_electroaimant()}
