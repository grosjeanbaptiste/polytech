"""Le theoreme de transport de Reynolds.

Chapitre III.6 des planches. La thermodynamique et la mecanique des
solides raisonnent sur un **systeme** : une quantite de matiere fixee,
dont la forme change. La mecanique des fluides raisonne sur un **volume
de controle**, que la matiere traverse. Le theoreme fait le lien.

Pour une grandeur extensive `B = b m`, le systeme et le volume de
controle coincidant a l'instant considere :

`dB_sys/dt = dB_CV/dt - b_1 rho_1 v_1 A_1 + b_2 rho_2 v_2 A_2`,

soit, pour un volume fixe et des entrees et sorties quelconques,
`d/dt (integrale de rho b dV) + integrale de rho b V . n dA`. La normale
`n` est sortante : `V . n` est positif en sortie, negatif en entree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Sequence

type Grandeur = sp.Expr | float
"""Un reel ou un symbole."""

Triplet = tuple[Grandeur, Grandeur, Grandeur]


def debit_de_grandeur(
    massique: Grandeur, masse_volumique: Grandeur, vitesse: Grandeur, aire: Grandeur
) -> Grandeur:
    """`B_point = b rho v A` : ce que l'ecoulement emporte par unite de temps."""
    return massique * masse_volumique * vitesse * aire


def flux_a_travers(
    massique: Grandeur,
    masse_volumique: Grandeur,
    vitesse: Triplet,
    normale: Triplet,
    aire: Grandeur,
) -> Grandeur:
    """`rho b (V . n) A` : positif en sortie, negatif en entree."""
    unitaire = sp.Matrix(normale)
    if sp.simplify(unitaire.dot(unitaire) - 1) != 0:
        msg = f"normale unitaire attendue, et non {normale}"
        raise ValueError(msg)
    normal = sp.Matrix(vitesse).dot(unitaire)
    return debit_de_grandeur(massique, masse_volumique, normal, aire)


def variation_du_systeme(
    dans_le_volume: Grandeur, entrees: Sequence[Grandeur], sorties: Sequence[Grandeur]
) -> Grandeur:
    """`dB_sys/dt = dB_CV/dt - somme B_in + somme B_out`."""
    return dans_le_volume - sum(entrees) + sum(sorties)
