"""Dimensionner un KHN et un Fleischer-Tow (chapitre 3, partie II).

Planches 50, 51 et 60. Les deux biquads se dimensionnent sur des valeurs
normalisees -- `R_1 = R_2 = 1`, `C_1 = C_2 = 1` pour le KHN -- puis se
ramenent a des composants par la mise a l'echelle des impedances.
"""

from __future__ import annotations

import sympy as sp


def rapports_du_khn(pulsation: sp.Expr, qualite: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """`(R_6 / R_5, R_4 / R_3)` pour des constantes de temps unite (planche 51).

    Etape 2 : `R_6 / R_5 = w_0^2`. Etape 3 :
    `R_4 / R_3 = Q (1 + R_6 / R_5) / racine(R_6 / R_5) - 1`, qui doit
    rester positif : `Q` ne peut pas descendre sous `w_0 / (1 + w_0^2)`.
    """
    w, q = sp.sympify(pulsation), sp.sympify(qualite)
    if not (w > 0 and q > 0):
        msg = f"pulsation ou facteur de qualite nul ou negatif : {w}, {q}"
        raise ValueError(msg)
    six_sur_cinq = w**2
    quatre_sur_trois = q * (1 + six_sur_cinq) / w - 1
    if not quatre_sur_trois > 0:
        msg = f"facteur de qualite trop faible pour ce montage : {q}"
        raise ValueError(msg)
    return six_sur_cinq, quatre_sur_trois


def gain_du_khn(r3: sp.Expr, r4: sp.Expr, r5: sp.Expr, r6: sp.Expr) -> sp.Expr:
    """`G = (R_4 / R_6) (R_5 + R_6) / (R_3 + R_4)` : il n'est pas libre (planche 50)."""
    return r4 / r6 * (r5 + r6) / (r3 + r4)


def regles_de_fleischer_tow(
    nature: str, gain: sp.Expr, *, r1: sp.Expr, r2: sp.Expr, r: sp.Expr
) -> tuple[sp.Expr, sp.Expr, sp.Expr]:
    """`(R_4, R_5, R_6)` selon le type de filtre et le gain (planche 60).

    L'infini dit qu'une branche est retiree. Le passe-bas et le
    passe-haut sortent inverses, le passe-bande non.
    """
    if not sp.sympify(gain) > 0:
        msg = f"gain nul ou negatif : {gain}"
        raise ValueError(msg)
    regles = {
        "passe-bas": (sp.oo, sp.oo, r2 / gain),
        "passe-haut": (r1 / gain, r / gain, sp.oo),
        "passe-bande": (r1 / gain, sp.oo, sp.oo),
    }
    if nature not in regles:
        msg = f"type de filtre inconnu : {nature!r}, attendu {sorted(regles)}"
        raise ValueError(msg)
    return regles[nature]
