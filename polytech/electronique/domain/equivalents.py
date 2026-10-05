"""Les accelerateurs d'analyse : diviseur generalise, Thevenin et Norton.

Planches du chapitre 1, partie III (planches 33 a 39).

Le **diviseur de tension generalise** ne suppose pas que le bas du pont
est a la masse : `V_out = Z_2 / (Z_1 + Z_2) (V_top - V_bottom) +
V_bottom`.

Tout dipole lineaire -- composants lineaires et sources independantes ou
commandees -- se remplace par une source de tension en serie avec une
impedance (**Thevenin**) ou par une source de courant en parallele avec
la meme impedance (**Norton**) : `Z_th = Z_no`, `V_th = I_no Z_no`.

L'equivalent garantit a la **charge** la meme relation entre courant et
tension. Il ne dit rien de l'interieur : la puissance qu'il dissipe
n'est pas celle du circuit d'origine.
"""

from __future__ import annotations

from dataclasses import dataclass

import sympy as sp


def diviseur_generalise(
    haut: object, bas: object, tension_du_haut: object, tension_du_bas: object = 0
) -> sp.Expr:
    """`V_out`, prise entre `Z_1` (vers le haut) et `Z_2` (vers le bas)."""
    z_1, z_2 = sp.sympify(haut), sp.sympify(bas)
    v_haut, v_bas = sp.sympify(tension_du_haut), sp.sympify(tension_du_bas)
    return z_2 / (z_1 + z_2) * (v_haut - v_bas) + v_bas


@dataclass(frozen=True, slots=True)
class EquivalentDeNorton:
    """Une source de courant `I_no` en parallele avec `Z_no`."""

    courant: sp.Expr
    impedance: sp.Expr

    def thevenin(self) -> EquivalentDeThevenin:
        """`V_th = I_no Z_no`, `Z_th = Z_no`."""
        return EquivalentDeThevenin(self.courant * self.impedance, self.impedance)


@dataclass(frozen=True, slots=True)
class EquivalentDeThevenin:
    """Une source de tension `V_th` en serie avec `Z_th`."""

    tension: sp.Expr
    impedance: sp.Expr

    def norton(self) -> EquivalentDeNorton:
        """`I_no = V_th / Z_th` ; leve si la source est ideale."""
        if self.impedance == 0:
            msg = "impedance de Thevenin nulle : la source est ideale, sans Norton"
            raise ValueError(msg)
        return EquivalentDeNorton(self.tension / self.impedance, self.impedance)

    def tension_sur(self, charge: object) -> sp.Expr:
        """La tension aux bornes d'une charge : un diviseur."""
        return diviseur_generalise(self.impedance, charge, self.tension)

    def puissance_interne(self, charge: object) -> sp.Expr:
        """`Z_th I^2`, ce que l'equivalent dissipe -- pas le circuit d'origine."""
        courant = self.tension / (self.impedance + sp.sympify(charge))
        return sp.simplify(self.impedance * courant**2)


def thevenin_du_diviseur(
    source: object, serie: object, shunt: object
) -> EquivalentDeThevenin:
    """Une source derriere `serie`, vue aux bornes de `shunt`."""
    z_serie, z_shunt = sp.sympify(serie), sp.sympify(shunt)
    ouverte = diviseur_generalise(z_serie, z_shunt, source)
    eteinte = z_serie * z_shunt / (z_serie + z_shunt)
    return EquivalentDeThevenin(sp.simplify(ouverte), sp.simplify(eteinte))
