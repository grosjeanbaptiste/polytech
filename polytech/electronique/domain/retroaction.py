"""Les parametres hybrides, et ce que la retroaction serie-parallele en fait.

La matrice H exprime `(V_1, I_2)` a partir de `(I_1, V_2)`. Les quatre
parametres sont donc les derivees partielles de ces deux grandeurs par
rapport a ces deux-la : construire la matrice comme un jacobien, c'est
appliquer la definition et rien d'autre.

Une retroaction serie-parallele additionne les matrices H de ses deux
quadripoles. C'est ce resultat du cours qui permet ensuite de lire le gain
en boucle fermee et l'impedance de sortie directement sur la matrice
globale.
"""

from __future__ import annotations

import sympy as sp


def _jacobien(
    tension_d_entree: sp.Expr,
    courant_de_sortie: sp.Expr,
    courant_d_entree: sp.Symbol,
    tension_de_sortie: sp.Symbol,
) -> sp.Matrix:
    """La matrice H, par definition : le jacobien de `(V_1, I_2)`."""
    return sp.Matrix(
        [
            [
                sp.simplify(sp.diff(tension_d_entree, courant_d_entree)),
                sp.simplify(sp.diff(tension_d_entree, tension_de_sortie)),
            ],
            [
                sp.simplify(sp.diff(courant_de_sortie, courant_d_entree)),
                sp.simplify(sp.diff(courant_de_sortie, tension_de_sortie)),
            ],
        ],
    )


def parametres_h_amplificateur(
    resistance_entree: sp.Expr,
    resistance_sortie: sp.Expr,
    gain_en_boucle_ouverte: sp.Expr,
) -> sp.Matrix:
    """H du circuit canonique : `R_in` a l'entree, `A V_1` en serie avec `R_o`."""
    if resistance_sortie == 0:
        msg = "resistance de sortie nulle : la maille de sortie n'a plus de loi"
        raise ValueError(msg)
    i1, v2 = sp.symbols("I_1 V_2")
    v1 = resistance_entree * i1
    i2 = (v2 - gain_en_boucle_ouverte * v1) / resistance_sortie
    return _jacobien(v1, i2, i1, v2)


def parametres_h_reseau_resistif(
    vers_la_masse: sp.Expr,
    de_contre_reaction: sp.Expr,
) -> sp.Matrix:
    """H du pont `R_1` a la masse et `R_2` en contre-reaction."""
    if vers_la_masse == 0 or de_contre_reaction == 0:
        msg = f"resistance nulle dans le pont : {vers_la_masse}, {de_contre_reaction}"
        raise ValueError(msg)
    i1, v2 = sp.symbols("I_1 V_2")
    parallele = (
        vers_la_masse * de_contre_reaction / (vers_la_masse + de_contre_reaction)
    )
    v1 = parallele * (i1 + v2 / de_contre_reaction)
    i2 = (v2 - v1) / de_contre_reaction
    return _jacobien(v1, i2, i1, v2)


def _rapport_de_boucle(matrice: sp.Matrix) -> sp.Expr:
    """`h_21 / (h_11 h_22)`, le nombre autour duquel tout le calcul tourne."""
    denominateur = matrice[0, 0] * matrice[1, 1]
    if denominateur == 0:
        msg = "matrice H degeneree : h_11 h_22 nul"
        raise ValueError(msg)
    return matrice[1, 0] / denominateur


def gain_en_boucle_fermee(matrice: sp.Matrix) -> sp.Expr:
    """Le gain en tension que la retroaction laisse, lu sur la matrice H."""
    boucle = _rapport_de_boucle(matrice)
    return -boucle / (1 - matrice[0, 1] * boucle)


def impedance_de_sortie_en_boucle_fermee(matrice: sp.Matrix) -> sp.Expr:
    """L'impedance de sortie que la retroaction laisse, lue sur la matrice H."""
    boucle = _rapport_de_boucle(matrice)
    return (1 / matrice[1, 1]) / (1 - matrice[0, 1] * boucle)
