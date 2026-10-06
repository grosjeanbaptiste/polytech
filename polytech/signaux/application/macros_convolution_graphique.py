"""Macros de la note « Convolution graphique ». Prefixe `ConvGraph`."""

from __future__ import annotations

import numpy as np

from polytech.signaux.application import convolution_graphique as cg
from polytech.signaux.application.formatage import coordonnees, nombre

_SOMMETS = np.array([-0.5, 0.0, 1.0, 2.0, 3.0, 3.5])
"""Les abscisses ou la sortie, affine par morceaux, change de pente."""


def macros_de_la_convolution() -> dict[str, str]:
    """Les sommets de la courbe, et les valeurs que le corrige cite."""
    ordonnees = np.array([float(cg.sortie(float(t))) for t in _SOMMETS])
    return {
        "ConvGraphSortie": coordonnees(_SOMMETS, ordonnees, 2),
        "ConvGraphSommet": nombre(float(cg.sortie(1)), 2),
        "ConvGraphCreux": nombre(float(cg.sortie(2)), 2),
        "ConvGraphAvantSommet": nombre(cg.aire_du_produit(0.6), 2),
        "ConvGraphBascule": nombre(cg.aire_du_produit(1.4), 2),
    }
