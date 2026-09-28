"""L'analyse nodale modifiee du filtre en echelle LC de l'enonce.

L'analyse nodale ordinaire ne sait pas traiter une source de tension : son
courant n'est pas une fonction des tensions de noeuds. L'analyse *modifiee*
le resout en ajoutant ce courant aux inconnues, et l'equation de la source
aux equations -- d'ou une matrice carree d'ordre quatre pour trois noeuds.

Le circuit : la source attaque le noeud 1, `R` mene au noeud 2, `L` au
noeud 3, ou `C` et `R` retournent a la masse.
"""

from __future__ import annotations

import sympy as sp


def systeme_en_echelle_lc(
    resistance: sp.Expr,
    inductance: sp.Expr,
    capacite: sp.Expr,
    variable: sp.Expr,
) -> tuple[sp.Matrix, sp.Matrix]:
    """Rend la matrice du systeme et son second membre.

    L'ordre des inconnues est celui impose par l'enonce :
    `[V_1(s), V_2(s), V_3(s), I_S(s)]`. Le second membre ne porte que la
    tension de source, sur la ligne de l'equation de source.
    """
    if resistance == 0 or inductance == 0 or capacite == 0:
        msg = f"composant nul dans l'echelle : {resistance}, {inductance}, {capacite}"
        raise ValueError(msg)
    conductance = 1 / resistance
    admittance_l = 1 / (variable * inductance)
    admittance_c = variable * capacite
    matrice = sp.Matrix(
        [
            [-conductance, conductance, 0, 1],
            [conductance, -(conductance + admittance_l), admittance_l, 0],
            [0, admittance_l, -(admittance_l + admittance_c + conductance), 0],
            [1, 0, 0, 0],
        ],
    )
    return matrice, sp.Matrix([0, 0, 0, sp.Symbol("V_S")])


def transfert_de_sortie(
    resistance: sp.Expr,
    inductance: sp.Expr,
    capacite: sp.Expr,
    variable: sp.Expr,
) -> sp.Expr:
    """`V_3(s) / V_S(s)`, obtenu en resolvant le systeme nodal modifie."""
    matrice, second_membre = systeme_en_echelle_lc(
        resistance,
        inductance,
        capacite,
        variable,
    )
    if matrice.det() == 0:
        msg = "systeme nodal singulier : le circuit n'a pas de solution unique"
        raise ValueError(msg)
    solution = matrice.solve(second_membre)
    return sp.cancel(sp.simplify(solution[2] / sp.Symbol("V_S")))


def systeme_avec_transconductance(
    resistance_de_source: sp.Expr,
    resistance_d_entree: sp.Expr,
    resistance_de_contre_reaction: sp.Expr,
    resistance_de_charge: sp.Expr,
    transconductance: sp.Expr,
) -> tuple[sp.Matrix, sp.Matrix]:
    """Le systeme nodal modifie de l'etage a source de courant commandee.

    La source `g_m V_2` n'ajoute aucune inconnue : son courant est une
    fonction d'une tension de noeud, donc elle se range dans la matrice
    comme une conductance de plus.
    """
    if resistance_de_contre_reaction == 0 or resistance_de_charge == 0:
        msg = (
            "resistance nulle dans l'etage : "
            f"{resistance_de_contre_reaction}, {resistance_de_charge}"
        )
        raise ValueError(msg)
    gs = 1 / resistance_de_source
    gin = 1 / resistance_d_entree
    gf = 1 / resistance_de_contre_reaction
    gc = 1 / resistance_de_charge
    matrice = sp.Matrix(
        [
            [-gs, gs, 0, 1],
            [gs, -(gs + gin + gf), gf, 0],
            [0, gf - transconductance, -(gf + gc), 0],
            [1, 0, 0, 0],
        ],
    )
    return matrice, sp.Matrix([0, 0, 0, sp.Symbol("V_S")])


def transfert_avec_transconductance(
    resistance_de_source: sp.Expr,
    resistance_d_entree: sp.Expr,
    resistance_de_contre_reaction: sp.Expr,
    resistance_de_charge: sp.Expr,
    transconductance: sp.Expr,
) -> sp.Expr:
    """`V_3 / V_S`, obtenu en resolvant le systeme nodal complet."""
    matrice, second_membre = systeme_avec_transconductance(
        resistance_de_source,
        resistance_d_entree,
        resistance_de_contre_reaction,
        resistance_de_charge,
        transconductance,
    )
    if matrice.det() == 0:
        msg = "systeme singulier : l'etage n'a pas de solution unique"
        raise ValueError(msg)
    return sp.cancel(matrice.solve(second_membre)[2] / sp.Symbol("V_S"))
