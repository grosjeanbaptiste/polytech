"""Cours III.3 -- pourquoi psi et phi n'existent pas toujours.

Les conditions d'existence ne sont pas des conventions : elles tombent du
theoreme de Schwarz. Un champ qui derive d'une fonction de courant est
*necessairement* a divergence nulle, et un champ qui derive d'un potentiel
est *necessairement* irrotationnel. C'est pourquoi la bibliotheque leve
quand la condition n'est pas remplie.
"""

from __future__ import annotations

import sympy as sp

from polytech.fluides.domain.cinematique.lignes import (
    fonction_de_courant,
    potentiel_des_vitesses,
)
from polytech.fluides.domain.cinematique.symboles import X, Y


def divergence_d_un_champ_derivant_de_psi() -> sp.Expr:
    """``div V`` pour ``u = dpsi/dy``, ``v = -dpsi/dx`` : nulle par Schwarz."""
    psi = sp.Function("psi")(X, Y)
    u, v = sp.diff(psi, Y), -sp.diff(psi, X)
    return sp.simplify(sp.diff(u, X) + sp.diff(v, Y))


def rotation_d_un_champ_derivant_de_phi() -> sp.Expr:
    """``omega_z`` pour ``V = grad phi`` : nul par Schwarz."""
    phi = sp.Function("phi")(X, Y)
    u, v = sp.diff(phi, X), sp.diff(phi, Y)
    return sp.simplify((sp.diff(v, X) - sp.diff(u, Y)) / 2)


def gradients_orthogonaux(u: sp.Expr, v: sp.Expr) -> sp.Expr:
    """``grad phi . grad psi`` : nul, d'ou le reseau orthogonal du trace."""
    phi = potentiel_des_vitesses(u, v)
    psi = fonction_de_courant(u, v)
    return sp.simplify(
        sp.diff(phi, X) * sp.diff(psi, X) + sp.diff(phi, Y) * sp.diff(psi, Y),
    )
