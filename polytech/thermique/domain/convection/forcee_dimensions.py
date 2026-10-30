"""L'analyse dimensionnelle de la convection forcee (Vaschy-Buckingham).

Sortie de `forcee.py`, qui les re-exporte : les noms restent importables
d'ou ils l'etaient.
"""

from __future__ import annotations

from typing import Final

VARIABLES_DE_LA_CONVECTION_FORCEE: Final = (
    "coefficient d'echange",
    "longueur caracteristique",
    "conductivite du fluide",
    "masse volumique",
    "viscosite dynamique",
    "chaleur massique",
    "vitesse",
)
"""Les sept variables que le fascicule denombre pour la convection forcee.

Quatre dimensions fondamentales les mesurent -- masse, longueur, temps,
temperature -- donc le theoreme de Vaschy-Buckingham annonce trois
groupes sans dimension. Ce sont Nusselt, Reynolds et Prandtl, et c'est
**pourquoi** toutes les correlations du chapitre ont cette forme-la.
"""

DIMENSIONS_FONDAMENTALES: Final = 4
"""Masse, longueur, temps, temperature."""


def nombre_de_groupes_de_vaschy_buckingham(variables: int, dimensions: int) -> int:
    """`n - k` : le nombre de groupes sans dimension independants.

    C'est l'etape que le fascicule detaille sur six planches, et elle
    repond a une question qu'on ne se pose pas assez : pourquoi deux
    nombres et non sept variables. Parce qu'il n'y a que `n - k` facons
    independantes de les combiner sans dimension.
    """
    if variables <= dimensions:
        msg = (
            f"moins de variables que de dimensions : {variables} <= {dimensions}, "
            f"aucun groupe ne peut se former"
        )
        raise ValueError(msg)
    return variables - dimensions


def groupes_sans_dimension() -> tuple[str, str, str]:
    """Les trois groupes que l'analyse dimensionnelle produit, dans l'ordre."""
    return "Nusselt", "Reynolds", "Prandtl"
