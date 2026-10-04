"""Macros du TP 2 d'electronique : des expressions, pas des nombres.

Le TP est litteral. Ses macros sont donc des expressions composees en
LaTeX, tirees des formules de la bibliotheque : les impedances, leurs
zeros et poles exacts, leurs limites en continu et a l'infini, les
transferts et les tensions de noeud. Le seul nombre est le volt du
premier diviseur.

Le prefixe est `TpDeux`.
"""

from __future__ import annotations

import sympy as sp

from polytech.electronique.application import circuits_tp_02 as c
from polytech.electronique.application.symboles_tp_02 import S

NOMS = {"2.1a": "Ua", "2.1b": "Ub", "2.2": "Deux", "2.3": "Trois"}


def _latex(expression: sp.Expr) -> str:
    """LaTeX avec `j` pour l'unite imaginaire, comme le cours l'ecrit."""
    return str(sp.latex(expression, imaginary_unit="j"))


def paire_de_racines(polynome: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """`-b/(2a)` et `b^2/(4a^2) - c/a` : centre et carre du demi-ecart."""
    a, b, c_ = sp.Poly(polynome, S).all_coeffs()
    return sp.simplify(-b / (2 * a)), sp.expand(b**2 / (4 * a**2) - c_ / a)


def _racines(expression: sp.Expr) -> str:
    poly = sp.Poly(expression, S)
    if poly.degree() == 0:
        return r"\varnothing"
    if poly.degree() == 1:
        return _latex(sp.solve(expression, S)[0])
    centre, carre = paire_de_racines(expression)
    return f"{_latex(centre)} \\pm \\sqrt{{{_latex(carre)}}}"


def _macros_des_impedances() -> dict[str, str]:
    macros: dict[str, str] = {}
    for cle, nom in NOMS.items():
        z = c.impedances_du_2()[cle]
        numerateur, denominateur = sp.fraction(sp.together(z))
        macros |= {
            f"TpDeuxZ{nom}": _latex(z),
            f"TpDeuxZeros{nom}": _racines(numerateur),
            f"TpDeuxPoles{nom}": _racines(denominateur),
            f"TpDeuxContinu{nom}": _latex(sp.limit(z, S, 0)),
        }
    return macros


def macros_du_tp_deux() -> dict[str, str]:
    """Les six sections du TP."""
    q = c.transferts_du_4_1()
    n = c.noeuds_du_5_2()
    d = c.contributions_du_5_3()
    h = c.transferts_du_6_1()
    return {
        **_macros_des_impedances(),
        "TpDeuxDiviseurContinu": str(q["continu"]),
        "TpDeuxDiviseurRC": _latex(sp.simplify(q["rc"])),
        "TpDeuxDiviseurCharge": _latex(sp.simplify(q["charge"])),
        "TpDeuxCinqUn": _latex(sp.factor(c.sortie_du_5_1())),
        "TpDeuxNoeudUn": _latex(n[1]),
        "TpDeuxNoeudDeux": _latex(n[2]),
        "TpDeuxNoeudTrois": _latex(n[3]),
        "TpDeuxTroisVolts": _latex(d["continu"]),
        "TpDeuxSousE": _latex(d["tension"]),
        "TpDeuxSousI": _latex(d["courant"]),
        "TpDeuxHa": _latex(sp.simplify(h["a"])),
        "TpDeuxHb": _latex(sp.simplify(h["b"])),
        "TpDeuxHc": _latex(sp.simplify(h["c"])),
    }
