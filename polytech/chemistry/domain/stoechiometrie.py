"""La stoechiometrie : ce qu'une equation equilibree permet de calculer.

Source : syllabus « Introduction aux sciences de l'ingenieur » 2025-2026,
paragraphe L.2.3, pages 102 et 103. Les coefficients sont ceux de l'equation
equilibree (`reactions.est_equilibree` le verifie) ; les quantites en moles.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Mapping


def _avancement(
    quantites: Mapping[str, float], coefficients: Mapping[str, int]
) -> dict[str, float]:
    """L'avancement que chaque reactif permettrait, seul : n / coefficient."""
    sans = [r for r in quantites if r not in coefficients]
    if sans:
        msg = f"reactif sans coefficient dans l'equation : {sans}"
        raise ValueError(msg)
    return {r: quantites[r] / coefficients[r] for r in quantites}


def reactif_limitant(
    quantites: Mapping[str, float], coefficients: Mapping[str, int]
) -> str:
    """Le premier reactif epuise : celui dont l'avancement est le plus petit."""
    avancements = _avancement(quantites, coefficients)
    return min(avancements, key=avancements.__getitem__)


def quantites_restantes(
    quantites: Mapping[str, float], coefficients: Mapping[str, int]
) -> dict[str, float]:
    """Les reactifs qui restent une fois la reaction menee au bout."""
    avancement = min(_avancement(quantites, coefficients).values())
    return {r: quantites[r] - coefficients[r] * avancement for r in quantites}


def produit_forme(quantite: float, coefficient: int, coefficient_produit: int) -> float:
    """Les moles de produit formees par la consommation de `quantite` d'un reactif."""
    return quantite * coefficient_produit / coefficient
