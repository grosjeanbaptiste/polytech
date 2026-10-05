"""La retroaction parallele-parallele : l'amplificateur inverseur par ses `Y`.

Syllabus, section 3.6.5. L'inverseur, prive de sa resistance d'entree
`R_1`, est un amplificateur dont la resistance `R_2` relie la sortie a
l'entree : les deux quadripoles sont en parallele a l'entree comme a la
sortie, et leurs matrices d'admittance s'additionnent,

    Y(1) = [[1/R_in, 0], [A_v/R_o, 1/R_o]],   Y(2) = [[1/R_2, -1/R_2], [-1/R_2, 1/R_2]].

L'impedance de sortie qui en resulte est si faible qu'une approximation
prise trop tot la fausse : elle se calcule exactement,

    Z_out = R_o (R_1 + R_2) / (R_o + R_1 + R_2 + A_v R_1),

et ne s'approche qu'a la fin, en `R_o / (1 + beta A_v)`. Le syllabus
imprime `(R_o // R_2) / (1 + R_1 A_v / (R_1 + R_2 + R_o))` : sa matrice
(3.14) y a neglige `R_o` devant `A_v R_2`, ce qui coute un millieme.
"""

from __future__ import annotations

import sympy as sp


def parametres_y_de_l_amplificateur(
    entree: sp.Expr, sortie: sp.Expr, gain: sp.Expr
) -> sp.Matrix:
    """`[[1/R_in, 0], [A_v/R_o, 1/R_o]]` : l'entree inverseuse commande `-A_v V_1`."""
    if sortie == 0 or entree == 0:
        msg = "resistance de sortie nulle ou d'entree nulle : admittance infinie"
        raise ValueError(msg)
    return sp.Matrix([[1 / entree, 0], [gain / sortie, 1 / sortie]])


def parametres_y_de_la_contre_reaction(retour: sp.Expr) -> sp.Matrix:
    """`[[1/R_2, -1/R_2], [-1/R_2, 1/R_2]]` : une resistance d'un acces a l'autre."""
    if retour == 0:
        msg = "resistance de contre-reaction nulle : les deux acces sont confondus"
        raise ValueError(msg)
    admittance = 1 / retour
    return sp.Matrix([[admittance, -admittance], [-admittance, admittance]])


def impedance_de_sortie_de_l_inverseur(
    sortie: sp.Expr, gain: sp.Expr, *, vers_l_entree: sp.Expr, retour: sp.Expr
) -> sp.Expr:
    """`R_o (R_1 + R_2) / (R_o + R_1 + R_2 + A_v R_1)`, entree a la masse.

    Exacte pour une resistance d'entree infinie : la sortie voit `R_1 +
    R_2` vers la masse, et `R_o` vers la source commandee `-A_v V_1`.
    """
    pont = vers_l_entree + retour
    return sortie * pont / (sortie + pont + gain * vers_l_entree)
