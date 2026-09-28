"""Les unites d'une boucle de retroaction, et ce qu'elles s'imposent.

Un gain de boucle est sans dimension : c'est ce qui permet d'ecrire
`1 + g beta` sans rien convertir. Des lors, si le chemin direct convertit
une tension en un courant, le chemin de retour convertit necessairement un
courant en une tension -- l'un en siemens, l'autre en ohms.

Le corrige de l'examen blanc donne une seule unite pour les deux, en la
justifiant par la conversion inverse de celle qu'elle decrit : c'est cette
confusion que ces trois fonctions rendent verifiable.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from sympy.physics.units import ampere, volt
from sympy.physics.units.util import quantity_simplify

if TYPE_CHECKING:
    import sympy as sp


def unite_de_la_transconductance() -> sp.Expr:
    """Ampere par volt : le chemin direct, `I_2 = g_ol V_1`."""
    return ampere / volt


def unite_de_la_transimpedance() -> sp.Expr:
    """Volt par ampere : le chemin de retour, `V_1 = beta I_2`."""
    return volt / ampere


def le_gain_de_boucle_est_sans_dimension(
    transconductance: sp.Expr,
    transimpedance: sp.Expr,
) -> bool:
    """Vrai si le produit des deux unites se simplifie en un pur nombre."""
    produit = quantity_simplify(transconductance * transimpedance)
    return bool(produit.is_number)
