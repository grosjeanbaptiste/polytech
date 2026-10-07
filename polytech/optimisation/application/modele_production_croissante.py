"""Modelisations particulieres, variante II : pourquoi aucun plan ne convient.

La planche 50 interdit toute baisse de production et impose de produire
exactement la demande totale. Les deux ne tiennent pas ensemble.
"""

from __future__ import annotations

import itertools

from polytech.optimisation.application.modele_production import DEMANDES


def plus_petit_total_croissant() -> int:
    """Le moins qu'un plan jamais decroissant doive produire pour servir a temps.

    Produire chaque mois la plus forte cadence moyenne exigee suffit :
    aucun mois d'un plan plus econome ne depasse ce total-la.
    """
    cumulees = tuple(itertools.accumulate(DEMANDES))
    cadence = max(-(-cumul // mois) for mois, cumul in enumerate(cumulees, start=1))
    borne = cadence * len(DEMANDES)
    croissants = itertools.combinations_with_replacement(
        range(borne + 1), len(DEMANDES)
    )
    return min(
        sum(plan)
        for plan in croissants
        if all(
            p >= c for p, c in zip(itertools.accumulate(plan), cumulees, strict=True)
        )
    )
