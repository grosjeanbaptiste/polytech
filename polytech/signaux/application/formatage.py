"""Mise en forme des nombres et des courbes pour le document.

Les nombres sont publies **bruts**, a point decimal : c'est la syntaxe
d'entree de siunitx, qui fait seul la virgule et l'espace des milliers.

Une courbe est publiee comme la liste `(x,y) (x,y) ...` que l'option
`coordinates` de `pgfplots` lit telle quelle. Elle ne passe donc jamais
par siunitx : ses nombres sont des coordonnees de trace, pas des valeurs
que le lecteur lit.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def nombre(valeur: float, decimales: int) -> str:
    """Formate avec ce nombre de decimales, zeros de fin retires."""
    if decimales < 0:
        msg = f"nombre de decimales negatif : {decimales}"
        raise ValueError(msg)
    rendu = f"{valeur:.{decimales}f}"
    rendu = rendu.rstrip("0").rstrip(".") if "." in rendu else rendu
    return "0" if rendu == "-0" else rendu


def coordonnees(abscisses: np.ndarray, ordonnees: np.ndarray, decimales: int) -> str:
    """`(x,y) (x,y) ...`, chaque nombre arrondi a ce rang."""
    if len(abscisses) != len(ordonnees):
        msg = f"longueurs differentes : {len(abscisses)} et {len(ordonnees)}"
        raise ValueError(msg)
    return " ".join(
        f"({nombre(float(x), decimales)},{nombre(float(y), decimales)})"
        for x, y in zip(abscisses, ordonnees, strict=True)
    )
