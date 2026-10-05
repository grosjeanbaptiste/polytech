"""Les grandeurs integrales d'une couche limite, pour un profil semblable.

Planches du chapitre IX. Un profil semblable s'ecrit `u / U_e = g(s)`
avec `s = y / delta(x)`. Trois mesures s'en tirent :

* l'epaisseur de deplacement `delta* = integrale de (1 - u / U_e) dy`,
  dont il faudrait deplacer la paroi pour que l'ecoulement sans couche
  limite porte le meme debit ;
* l'epaisseur de quantite de mouvement
  `theta = integrale de (u / U_e)(1 - u / U_e) dy` ;
* le frottement parietal `tau_w = mu du/dy` en `y = 0`, et son
  coefficient `C_f = tau_w / (rho U_e^2 / 2)`.

Les deux premieres sont rendues en fraction de `delta`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import sympy as sp

if TYPE_CHECKING:
    from collections.abc import Callable


def _exiger_un_profil(profil: sp.Expr, variable: sp.Symbol) -> None:
    if profil.subs(variable, 0) != 0:
        msg = "condition de non-glissement violee : le profil ne part pas de zero"
        raise ValueError(msg)
    if profil.subs(variable, 1) != 1:
        msg = "le profil ne rejoint pas la vitesse exterieure au bord de la couche"
        raise ValueError(msg)


def deplacement_reduit(profil: sp.Expr, variable: sp.Symbol) -> sp.Expr:
    """`delta* / delta = integrale de 0 a 1 de (1 - g)`."""
    _exiger_un_profil(profil, variable)
    return sp.integrate(1 - profil, (variable, 0, 1))


def quantite_de_mouvement_reduite(profil: sp.Expr, variable: sp.Symbol) -> sp.Expr:
    """`theta / delta = integrale de 0 a 1 de g (1 - g)`."""
    _exiger_un_profil(profil, variable)
    return sp.integrate(profil * (1 - profil), (variable, 0, 1))


def frottement_parietal(
    profil: sp.Expr, variable: sp.Symbol, *, viscosite: float, vitesse: float
) -> Callable[[float], float]:
    """`tau_w = mu U_e g'(0) / delta`, en fonction de l'epaisseur `delta`."""
    _exiger_un_profil(profil, variable)
    pente = float(sp.diff(profil, variable).subs(variable, 0))

    def contrainte(epaisseur: float) -> float:
        if epaisseur <= 0:
            msg = f"epaisseur nulle ou negative : {epaisseur} m"
            raise ValueError(msg)
        return viscosite * vitesse * pente / epaisseur

    return contrainte


def coefficient_de_frottement(
    contrainte: float, masse_volumique: float, vitesse: float
) -> float:
    """`C_f = tau_w / (rho U_e^2 / 2)`."""
    if vitesse <= 0:
        msg = f"vitesse nulle ou negative : {vitesse} m/s"
        raise ValueError(msg)
    return contrainte / (masse_volumique * vitesse**2 / 2)
