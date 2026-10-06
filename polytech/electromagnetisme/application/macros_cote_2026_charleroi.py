"""Macros de l'exercice cote du 7 mai 2026 (Charleroi). Prefixe `CoteVingtSix`."""

from __future__ import annotations

from polytech.electromagnetisme.application import cote_2026_couplage as cg
from polytech.electromagnetisme.application import cote_2026_spire as sr
from polytech.electromagnetisme.application import cote_2026_thevenin as th
from polytech.electromagnetisme.application.formatage import (
    expression_latex,
    nombre,
)


def _thevenin() -> dict[str, str]:
    tension, resistance = th.thevenin()
    oubli_tension, oubli_resistance = th.thevenin(avec_la_bobine=False)
    premier, second = th.courants_de_maille()
    return {
        "CoteVingtSixTension": nombre(float(tension), 2),
        "CoteVingtSixResistance": nombre(float(resistance), 2),
        "CoteVingtSixCourantUn": nombre(float(premier), 2),
        "CoteVingtSixCourantDeux": nombre(float(second), 2),
        "CoteVingtSixCourtCircuit": nombre(float(tension / resistance), 2),
        "CoteVingtSixOubliTension": nombre(float(oubli_tension), 2),
        "CoteVingtSixOubliResistance": nombre(float(oubli_resistance), 2),
    }


def _spire() -> dict[str, str]:
    distance = sr.distance_de_la_force_maximale()
    return {"CoteVingtSixDistance": expression_latex(distance)}


def _couplage() -> dict[str, str]:
    return {
        "CoteVingtSixProduit": expression_latex(cg.produit_a_cinquante_hertz()),
        "CoteVingtSixCouplage": expression_latex(cg.mutuelle()),
        "CoteVingtSixCouplageMilli": nombre(float(cg.mutuelle()) * 1e3, 1),
        "CoteVingtSixRamenee": nombre(float(cg.resistance_ramenee()), 2),
    }


def macros_cote_vingt_six() -> dict[str, str]:
    """Les macros des trois questions."""
    return {**_thevenin(), **_spire(), **_couplage()}
